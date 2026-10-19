from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.deps import get_vector_store_service


def test_health_check_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["redis"] == "ok"


def test_document_upload_invalid_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")},
        data={"chunking_strategy": "fixed"},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "not allowed" in data["error"]["message"].lower()


def test_chat_endpoint_query(client: TestClient) -> None:
    mock_vector_store = AsyncMock()
    mock_vector_store.search.return_value = []

    client.app.dependency_overrides[get_vector_store_service] = lambda: mock_vector_store

    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": "test-chat-session",
            "message": "What is the document ingestion pipeline?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
