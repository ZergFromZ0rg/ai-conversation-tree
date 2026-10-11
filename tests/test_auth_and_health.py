def test_healthz_is_public_and_open_by_default(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/conversations").status_code == 200


def test_token_gates_the_api_but_not_the_shell(client, monkeypatch):
    monkeypatch.setenv("AI_CONVERSATION_TREE_API_TOKEN", "s3cret")
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/conversations").status_code == 401
    assert client.get("/conversations", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/conversations", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post("/ingest/conversations", json={}).status_code == 401


def test_dev_server_origin_passes_cors_preflight(client):
    response = client.request(
        "OPTIONS",
        "/ingest/conversations",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.request(
        "OPTIONS",
        "/ingest/conversations",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in other.headers
