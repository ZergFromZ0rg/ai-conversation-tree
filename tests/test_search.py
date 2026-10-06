from conftest import chat


def seed(client):
    ids = {}
    ids["cats"] = client.post(
        "/ingest/conversations",
        json={
            "source": "chatgpt",
            "externalId": "cats",
            "title": "Cats",
            "messages": chat(
                ("What are cats?", "Cats are small carnivorous mammals."),
                ("What do cats eat?", "Cats are obligate carnivores and need meat."),
            ),
        },
    ).json()["conversationId"]
    ids["sql"] = client.post(
        "/ingest/conversations",
        json={
            "source": "claude",
            "externalId": "sql",
            "title": "Databases",
            "messages": chat(
                ("What is a database index?", "An index is a structure that speeds up lookups on a column."),
            ),
        },
    ).json()["conversationId"]
    return ids


def test_search_ranks_the_matching_conversation_first(client):
    ids = seed(client)
    results = client.get("/search", params={"query": "cat food"}).json()["results"]
    assert results[0]["conversationId"] == ids["cats"]


def test_hit_carries_snippet_terms_and_position(client):
    ids = seed(client)
    body = client.get("/search", params={"query": "obligate carnivores"}).json()
    top = body["results"][0]
    assert top["conversationId"] == ids["cats"] and top["turnId"] == 1
    assert top["matchIn"] == "answer"
    assert "obligate" in top["matchedTerms"]
    assert "obligate" in top["snippet"].lower()
    assert top["turnCount"] == 2
    assert top["source"] == "chatgpt"
    assert body["terms"] == ["obligate", "carnivores"]


def test_question_hits_are_labelled_as_questions(client):
    seed(client)
    top = client.get("/search", params={"query": "database index"}).json()["results"][0]
    assert top["matchIn"] == "question"


def test_conversation_filter_scopes_results(client):
    ids = seed(client)
    results = client.get("/search", params={"query": "index", "conversationId": ids["cats"]}).json()["results"]
    assert all(result["conversationId"] == ids["cats"] for result in results)


def test_exact_words_find_turns_the_embedding_would_miss(client):
    ids = seed(client)
    results = client.get("/search", params={"query": "speeds"}).json()["results"]
    assert any(result["conversationId"] == ids["sql"] for result in results)


def test_search_validation(client):
    assert client.get("/search", params={"query": "  "}).status_code == 422
    assert client.get("/search", params={"query": "x", "limit": 0}).status_code == 422
    assert client.get("/search", params={"query": "x", "limit": 101}).status_code == 422
