import numpy as np

from conversation_tree.contextService import analyzeContextHealth, estimateTokens, recentHistory


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def turn(index, user="question", ai="answer"):
    return {"id": index, "userText": user, "aiText": ai}


def orthogonal(count):
    return np.eye(count, dtype=np.float32)


def test_clean_conversation_has_no_flags():
    turns = [turn(i, f"topic {i}") for i in range(4)]
    report = analyzeContextHealth(turns, orthogonal(4), [])
    assert report["overall"] == "low"
    assert report["flaggedTurns"] == []
    assert report["horizonTurnId"] is None


def test_repeated_question_is_flagged_as_reask():
    vectors = orthogonal(4)
    vectors[3] = vectors[0]  # turn 3 asks what turn 0 asked
    turns = [turn(i, f"topic {i}") for i in range(4)]
    report = analyzeContextHealth(turns, vectors, [])
    (flag,) = report["flaggedTurns"]
    assert flag["turnId"] == 3
    assert flag["signals"][0]["kind"] == "reask"
    assert flag["signals"][0]["turnId"] == 0


def test_immediate_repeat_needs_near_verbatim_similarity():
    nearly = np.stack([unit(1, 0), unit(1, 0.6)])  # cosine ~0.86
    turns = [turn(0), turn(1)]
    assert analyzeContextHealth(turns, nearly, [])["flaggedTurns"] == []
    verbatim = np.stack([unit(1, 0), unit(1, 0.1)])  # cosine ~0.995
    assert analyzeContextHealth(turns, verbatim, [])["flaggedTurns"][0]["turnId"] == 1


def test_memory_cue_is_flagged_but_not_on_the_first_turn():
    turns = [turn(0, "As I said, use tabs"), turn(1, "You forgot that I already told you to use tabs")]
    report = analyzeContextHealth(turns, orthogonal(2), [])
    assert [item["turnId"] for item in report["flaggedTurns"]] == [1]
    assert report["flaggedTurns"][0]["signals"][0]["kind"] == "memoryCue"


def test_far_reach_beyond_the_window_is_high_risk():
    long = "x" * 4000  # ~1000 tokens per message
    turns = [turn(0, "setup", "ok")] + [turn(i, long, long) for i in range(1, 6)] + [turn(6, "back to setup")]
    edges = [{"fromTurnId": 0, "toTurnId": 6, "label": "related"}]
    report = analyzeContextHealth(turns, orthogonal(7), edges, windowTokens=3000)
    flag = next(item for item in report["flaggedTurns"] if item["turnId"] == 6)
    assert flag["level"] == "high"
    assert flag["signals"][0]["kind"] == "farReach"
    assert "outside the model's window" in flag["signals"][0]["detail"]


def test_far_reach_inside_the_window_is_only_long_range():
    turns = [turn(i) for i in range(6)]
    edges = [{"fromTurnId": 0, "toTurnId": 5, "label": "continuation"}]
    report = analyzeContextHealth(turns, orthogonal(6), edges, windowTokens=100000)
    flag = report["flaggedTurns"][0]
    assert flag["signals"][0]["kind"] == "longRange"
    assert flag["level"] == "low"


def test_nearby_references_are_not_flagged():
    turns = [turn(i) for i in range(6)]
    edges = [{"fromTurnId": 3, "toTurnId": 5, "label": "continuation"}]
    assert analyzeContextHealth(turns, orthogonal(6), edges)["flaggedTurns"] == []


def test_horizon_marks_where_the_window_starts():
    big = "x" * 4000  # 1000 tokens each side -> 2000 per turn
    turns = [turn(i, big, big) for i in range(6)]
    report = analyzeContextHealth(turns, orthogonal(6), [], windowTokens=5000)
    # the newest two turns (4000 tokens) fit; a third (6000) does not
    assert report["horizonTurnId"] == 4
    assert report["windowPressure"] == round(12000 / 5000, 2)


def test_empty_conversation():
    report = analyzeContextHealth([], np.zeros((0, 1), dtype=np.float32), [])
    assert report["overall"] == "low" and report["turnCount"] == 0


def test_token_estimate_is_never_zero():
    assert estimateTokens("") == 1


# --- endpoint -------------------------------------------------------------

def test_context_health_endpoint(client):
    from conftest import chat

    body = client.post(
        "/ingest/conversations",
        json={
            "source": "claude",
            "externalId": "ctx",
            "messages": chat(
                ("How do I center a div?", "Use flexbox with justify-content and align-items."),
                ("What is the capital of France?", "Paris."),
                ("How do I center a div?", "Use flexbox with justify-content and align-items."),
            ),
        },
    ).json()
    health = client.get(f"/conversations/{body['conversationId']}/context-health").json()
    assert any(item["turnId"] == 2 and item["signals"][0]["kind"] == "reask" for item in health["flaggedTurns"])
    assert client.get("/conversations/9999/context-health").status_code == 404


# --- recentHistory: what a model is sent along with a new message ---------------

def timelineTurn(label):
    """A turn costing exactly 20 tokens (40 characters each way)."""
    return {"userText": label * 40, "aiText": label.upper() * 40}


def contents(messages):
    return [message["content"][0] for message in messages]


NEW = "x" * 8  # 2 tokens


def test_history_is_chat_messages_oldest_first():
    messages = recentHistory([timelineTurn("a"), timelineTurn("b")], NEW, windowTokens=1000)
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert contents(messages) == ["a", "A", "b", "B"]


def test_the_oldest_turns_are_dropped_first():
    turns = [timelineTurn(letter) for letter in "abcde"]
    assert contents(recentHistory(turns, NEW, windowTokens=62)) == ["c", "C", "d", "D", "e", "E"]


def test_a_turn_that_exactly_fits_is_kept():
    turns = [timelineTurn(letter) for letter in "abc"]
    assert len(recentHistory(turns, NEW, windowTokens=62)) == 6  # 3 turns = 60 tokens = the budget
    assert len(recentHistory(turns, NEW, windowTokens=61)) == 4  # one token short drops the oldest whole turn


def test_the_new_message_is_charged_against_the_window_first():
    turns = [timelineTurn("a"), timelineTurn("b")]
    assert contents(recentHistory(turns, NEW, windowTokens=22)) == ["b", "B"]
    assert recentHistory(turns, NEW, windowTokens=21) == []
    assert recentHistory(turns, "y" * 400, windowTokens=50) == [], "a message bigger than the window leaves no room"


def test_history_is_contiguous_it_never_skips_a_big_turn_to_reach_older_ones():
    big = {"userText": "B" * 800, "aiText": "B" * 800}
    turns = [timelineTurn("a"), big, timelineTurn("c")]
    assert contents(recentHistory(turns, NEW, windowTokens=100)) == ["c", "C"]


def test_no_turns_means_no_history():
    assert recentHistory([], NEW, windowTokens=1000) == []
