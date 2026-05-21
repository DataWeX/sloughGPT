"""
Streaming Inference for Real-Time Generation

Provides streaming token-by-token generation backed by InferenceEngine.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Iterator, List, Optional
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """A single token in the stream."""
    token: str
    token_id: int
    logprob: float
    is_final: bool
    metadata: Dict[str, Any]


class StreamingGenerator:
    """
    Streaming token generator for real-time output.

    Backed by InferenceEngine for real model inference with KV cache,
    sampling, and latency tracking.

    Features:
    - Token-by-token streaming via InferenceEngine
    - Configurable output format
    - Latency tracking
    - Backpressure handling
    """

    def __init__(self, engine=None):
        self._engine = engine
        self.tokens_generated = 0
        self.start_time = 0.0

    def set_engine(self, engine):
        """Set the InferenceEngine to use for generation."""
        self._engine = engine

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        on_token: Optional[Callable[[StreamChunk], None]] = None,
    ) -> Iterator[StreamChunk]:
        """
        Generate tokens as a stream using InferenceEngine.

        Yields StreamChunk objects as tokens are generated.
        Falls back to simulated token-by-token output if no engine is set.
        """
        self.start_time = time.time()
        self.tokens_generated = 0

        if self._engine is not None:
            token_id = 0
            async for token_text in self._engine.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                chunk = StreamChunk(
                    token=token_text,
                    token_id=token_id,
                    logprob=0.0,
                    is_final=False,
                    metadata={
                        "tokens_per_second": self.tokens_generated / max(time.time() - self.start_time, 0.001),
                        "latency_ms": (time.time() - self.start_time) * 1000,
                    }
                )

                self.tokens_generated += 1
                token_id += 1

                if on_token:
                    on_token(chunk)

                yield chunk
        else:
            words = prompt.split()
            for i, word in enumerate(words[:max_tokens]):
                await asyncio.sleep(0.01)

                chunk = StreamChunk(
                    token=word + " ",
                    token_id=i,
                    logprob=-0.1 * i,
                    is_final=(i == min(len(words), max_tokens) - 1),
                    metadata={
                        "tokens_per_second": self.tokens_generated / max(time.time() - self.start_time, 0.001),
                        "latency_ms": (time.time() - self.start_time) * 1000,
                    }
                )

                self.tokens_generated += 1

                if on_token:
                    on_token(chunk)

                yield chunk

    def generate_stream_sync(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming (for non-async contexts)."""
        if self._engine is not None:
            import threading
            result_queue: List[StreamChunk] = []
            event = threading.Event()

            async def _run():
                async for chunk in self.generate_stream(prompt, max_tokens):
                    result_queue.append(chunk)
                event.set()

            thread = threading.Thread(target=lambda: asyncio.run(_run()))
            thread.start()
            event.wait()
            yield from result_queue
        else:
            words = prompt.split()
            for i, word in enumerate(words[:max_tokens]):
                yield StreamChunk(
                    token=word + " ",
                    token_id=i,
                    logprob=-0.1 * i,
                    is_final=(i == min(len(words), max_tokens) - 1),
                    metadata={}
                )

    def get_stats(self) -> Dict[str, float]:
        """Get generation statistics."""
        elapsed = time.time() - self.start_time
        return {
            "tokens_generated": self.tokens_generated,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.tokens_generated / max(elapsed, 0.001),
        }


class StreamingRAG:
    """RAG with streaming output backed by InferenceEngine."""

    def __init__(self, rag, streaming_generator: StreamingGenerator):
        self.rag = rag
        self.generator = streaming_generator

    async def stream_answer(
        self,
        question: str,
        max_tokens: int = 200,
    ) -> Iterator[StreamChunk]:
        """Stream RAG answer token by token."""
        results = self.rag.query(question)
        context = results.get("context", "")

        prompt = f"Based on: {context}\n\nQuestion: {question}\n\nAnswer:"

        async for chunk in self.generator.generate_stream(prompt, max_tokens):
            yield chunk


class StreamAggregator:
    """Aggregates streaming chunks into complete response."""

    def __init__(self):
        self.chunks: List[StreamChunk] = []

    def add(self, chunk: StreamChunk):
        """Add a chunk to the aggregator."""
        self.chunks.append(chunk)

    def get_full_text(self) -> str:
        """Get complete text from all chunks."""
        return "".join(c.token for c in self.chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        if not self.chunks:
            return {}

        avg_logprob = sum(c.logprob for c in self.chunks) / len(self.chunks)
        final_chunk = self.chunks[-1]

        return {
            "total_tokens": len(self.chunks),
            "text": self.get_full_text(),
            "avg_logprob": avg_logprob,
            "latency_ms": final_chunk.metadata.get("latency_ms", 0),
            "tokens_per_second": final_chunk.metadata.get("tokens_per_second", 0),
            "is_complete": final_chunk.is_final,
        }


async def demo_streaming():
    """Demonstrate streaming generation (fallback simulated mode)."""
    generator = StreamingGenerator()

    print("=" * 60)
    print("STREAMING GENERATION DEMO")
    print("=" * 60)
    print()

    aggregator = StreamAggregator()

    async for chunk in generator.generate_stream(
        "The quick brown fox jumps over the lazy dog",
        max_tokens=20,
    ):
        aggregator.add(chunk)

        stats = chunk.metadata
        print(f"Token: '{chunk.token.strip()}' (id={chunk.token_id}, logprob={chunk.logprob:.2f})")

    print()
    print("=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    final_stats = aggregator.get_stats()
    print(f"Total tokens: {final_stats['total_tokens']}")
    print(f"Full text: {final_stats['text']}")
    print(f"Tokens/sec: {final_stats['tokens_per_second']:.2f}")
    print(f"Latency: {final_stats['latency_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(demo_streaming())


__all__ = [
    "StreamChunk",
    "StreamingGenerator",
    "StreamingRAG",
    "StreamAggregator",
]
