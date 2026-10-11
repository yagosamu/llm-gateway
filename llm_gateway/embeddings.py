"""Text embeddings for the semantic cache, on OpenAI's embeddings endpoint.

Vectors come back L2-normalized as float32, so cosine similarity is a dot product. The token count
of each call is returned for cost accounting; an embedding has no output tokens."""
import datetime
import os
import time
from dataclasses import dataclass

import numpy as np
import openai

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
PRICE_PER_MTOK = 0.02  # US$, input only
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/text-embedding-3-small"
PRICE_CHECKED = datetime.date(2026, 10, 11)
MAX_INPUT_TOKENS = 8192  # per text; the largest deduplicated Dolly user message is about 23.6 kB
TIMEOUT_SECONDS = 30.0


def cost_usd(tokens: int) -> float:
    return tokens * PRICE_PER_MTOK / 1_000_000


def ceiling_usd(texts: list[str]) -> float:
    """Worst case: a token is at least one UTF-8 byte."""
    return cost_usd(sum(len(t.encode("utf-8")) for t in texts))


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@dataclass(frozen=True)
class Embedding:
    vectors: np.ndarray  # (n, DIMENSIONS), float32, unit length
    tokens: int
    seconds: float


class OpenAIEmbedder:
    def __init__(self, client: openai.AsyncOpenAI | None = None, model: str = MODEL):
        self.client = client or openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT_SECONDS,
                                                   max_retries=0)
        self.model = model

    async def embed(self, texts: list[str]) -> Embedding:
        started = time.perf_counter()
        response = await self.client.embeddings.create(model=self.model, input=texts)
        seconds = time.perf_counter() - started
        ordered = sorted(response.data, key=lambda d: d.index)
        return Embedding(normalize([d.embedding for d in ordered]), response.usage.prompt_tokens, seconds)
