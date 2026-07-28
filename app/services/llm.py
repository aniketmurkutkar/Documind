import hashlib

import httpx
from cachetools import TTLCache


class LLMService:
    def __init__(
        self,
        api_key: str,
        model: str,
        cache_ttl_seconds: int,
        cache_max_size: int,
        base_url: str = "https://api.openai.com/v1",
        llm_route: str = "chat_completions",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.llm_route = (
            llm_route.strip().lower()
            if llm_route.strip().lower() in ("chat_completions", "responses")
            else "chat_completions"
        )
        self.cache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl_seconds)

    def _cache_key(self, query: str, context: str) -> str:
        raw = f"{self.llm_route}||{query}||{context}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def route_query(self, query: str) -> str:
        """Return a fast/deep label from query length (placeholder for future model routing)."""
        token_count = len(query.split())
        return "fast" if token_count < 18 else "deep"

    async def answer(self, query: str, context: str) -> tuple[str, bool, str]:
        route = self.route_query(query)
        key = self._cache_key(query, context)
        if key in self.cache:
            return self.cache[key], True, route

        if not self.api_key:
            fallback = (
                "No LLM API key configured. Returning retrieved context only.\n\n"
                f"Question: {query}\n\nContext:\n{context[:1200]}"
            )
            return fallback, False, route

        prompt = (
            "You are an assistant for technical document QA.\n"
            "Answer only from the provided context. If context is insufficient, say so.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n"
        )

        headers = {"Authorization": f"Bearer {self.api_key}"}

        if self.llm_route == "responses":
            payload = {"model": self.model, "input": prompt}
            url = f"{self.base_url}/responses"
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You answer questions using only the provided context. "
                            "If the answer is not in the context, say you do not have enough information."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            try:
                err = response.json()
                detail = err.get("error", {}).get("message") or response.text
            except Exception:
                detail = response.text or response.reason_phrase
            msg = (
                f"LLM request failed ({response.status_code}). "
                f"{(detail or '')[:400]}"
            )
            return msg, False, route

        data = response.json()

        if self.llm_route == "responses":
            text = (data.get("output_text") or "").strip()
        else:
            choices = data.get("choices") or []
            if not choices:
                text = ""
            else:
                msg = choices[0].get("message") or {}
                text = (msg.get("content") or "").strip()
        if not text:
            text = "Model returned an empty response."
        self.cache[key] = text
        return text, False, route
