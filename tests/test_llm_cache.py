import asyncio

import pytest

import app.services.llm as llm_module
import app.services.pipeline as pipeline_module
from app.config import Settings
from app.services.llm import LLMService
from app.services.pipeline import RagPipeline

QUERY = "what does ERR_4021 mean"
CONTEXT = "Error ERR_4021 is raised when disk quota runs out"


class _StubEmbeddings:
    def __init__(self, model_name: str) -> None:
        pass


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str) -> None:
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient and records every upstream call."""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(f"answer #{len(_FakeAsyncClient.calls)}")


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient.calls


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "EmbeddingService", _StubEmbeddings)
    settings = Settings(
        _env_file=None,
        llm_api_key="",
        openai_api_key="",
        groq_api_key="",
        llm_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        llm_route="chat_completions",
    )
    return RagPipeline(settings)


def _byok_answer(pipeline: RagPipeline, api_key: str, model: str | None = None):
    llm = pipeline._llm_for_request(
        client_api_key=api_key,
        client_base_url=None,
        client_model=model,
        client_llm_route=None,
    )
    return asyncio.run(llm.answer(query=QUERY, context=CONTEXT))


def test_repeated_byok_query_is_served_from_cache(pipeline, fake_http):
    first, first_cached, _ = _byok_answer(pipeline, "sk-user-a")
    second, second_cached, _ = _byok_answer(pipeline, "sk-user-a")

    assert first_cached is False
    assert second_cached is True
    assert second == first
    assert len(fake_http) == 1


def test_cache_is_shared_across_api_keys(pipeline, fake_http):
    _byok_answer(pipeline, "sk-user-a")
    _, cached, _ = _byok_answer(pipeline, "sk-user-b")

    assert cached is True
    assert len(fake_http) == 1


def test_different_model_misses_cache(pipeline, fake_http):
    _byok_answer(pipeline, "sk-user-a", model="gpt-4o-mini")
    _, cached, _ = _byok_answer(pipeline, "sk-user-a", model="gpt-4o")

    assert cached is False
    assert len(fake_http) == 2
    assert fake_http[1]["json"]["model"] == "gpt-4o"


def test_cache_key_covers_base_url_model_and_route():
    def key(**overrides):
        kwargs = dict(
            api_key="sk-x",
            model="m1",
            cache_ttl_seconds=60,
            cache_max_size=10,
            base_url="https://a.example/v1",
            llm_route="chat_completions",
        )
        kwargs.update(overrides)
        return LLMService(**kwargs)._cache_key(QUERY, CONTEXT)

    base = key()
    assert key(api_key="sk-other") == base
    assert key(model="m2") != base
    assert key(base_url="https://b.example/v1") != base
    assert key(llm_route="responses") != base


def test_failed_llm_call_is_not_cached(pipeline, monkeypatch):
    class _ErrorClient(_FakeAsyncClient):
        async def post(self, url, headers=None, json=None):
            resp = _FakeResponse("")
            resp.status_code = 500
            resp.json = lambda: {"error": {"message": "upstream down"}}
            return resp

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _ErrorClient)
    _, cached, _ = _byok_answer(pipeline, "sk-user-a")

    assert cached is False
    assert len(pipeline.llm_cache) == 0
