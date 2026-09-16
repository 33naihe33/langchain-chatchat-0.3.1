from fastapi.testclient import TestClient
import numpy as np

from bge_m3_service import create_app


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(8193 if text == "too long" else len(text)))


class FakeModel:
    def encode(self, texts, return_dense=True):
        return {"dense_vecs": [[float(index), 1.0] for index, _ in enumerate(texts)]}


def make_client():
    return TestClient(create_app(lambda: (FakeModel(), FakeTokenizer())))


def test_embeddings_return_dense_vectors_in_input_order():
    response = make_client().post(
        "/v1/embeddings", json={"model": "bge-m3", "input": ["资金", "头寸"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "bge-m3"
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert [item["embedding"] for item in body["data"]] == [[0.0, 1.0], [1.0, 1.0]]


def test_over_limit_input_reports_actual_token_count():
    response = make_client().post(
        "/v1/embeddings", json={"model": "bge-m3", "input": "too long"}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "message": "input exceeds BGE-M3 limit; lower CHUNK_SIZE",
        "actual_tokens": 8193,
        "limit_tokens": 8192,
    }


def test_embeddings_serialize_numpy_float32_values():
    class NumpyModel:
        def encode(self, texts, return_dense=True):
            return {"dense_vecs": [np.array([np.float32(0.5)]) for _ in texts]}

    response = TestClient(
        create_app(lambda: (NumpyModel(), FakeTokenizer()))
    ).post("/v1/embeddings", json={"model": "bge-m3", "input": "资金"})

    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.5]
