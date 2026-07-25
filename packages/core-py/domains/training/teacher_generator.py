"""
Teacher Data Generator

Generates training pairs from a teacher model (HuggingFace) for student distillation.
Uses vector store for context retrieval — the teacher queries the loaded dataset
to find relevant passages before generating each response.

Pipeline:
    source_text → load into vector store → for each prompt:
        query store for relevant context → teacher generates with context → pairs → student trains

No differentiability required — the teacher just generates text.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("slo.training.teacher_generator")


def _chunk_text(text: str, max_chunk: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for storage and prompt generation.

    Args:
        text: Raw source text.
        max_chunk: Maximum characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chunk:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)
        start += max_chunk - overlap

    return chunks


def load_dataset_to_store(
    source_text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> Tuple[Any, List[str]]:
    """Load source text into an InMemoryVectorStore for context retrieval.

    Chunks the text, embeds each chunk via simple_embed, and inserts into
    the store. Returns the store and the raw chunks for direct access.

    Args:
        source_text: Raw dataset text.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        (vector_store, chunks) — the populated store and raw chunk list.
    """
    from domains.inference.vector_store import InMemoryVectorStore, VectorEntry, simple_embed

    chunks = _chunk_text(source_text, max_chunk=chunk_size, overlap=chunk_overlap)
    if not chunks:
        logger.warning("No chunks from source text (len=%d)", len(source_text))
        return InMemoryVectorStore(dimension=384), []

    store = InMemoryVectorStore(dimension=384)

    entries = []
    for i, chunk in enumerate(chunks):
        vec = simple_embed(chunk, dimension=384)
        entry = VectorEntry(
            id=f"chunk_{i}",
            vector=vec,
            text=chunk,
            metadata={"index": i, "length": len(chunk)},
        )
        entries.append(entry)

    store.upsert_sync(entries)

    logger.info(
        "Loaded %d chunks into vector store (source_len=%d)",
        len(chunks), len(source_text),
        extra={"tag": "TRAIN"},
    )

    return store, chunks


def retrieve_context(
    store: Any,
    query: str,
    top_k: int = 3,
    min_score: float = 0.1,
) -> str:
    """Query the vector store and return relevant context as a single string.

    Args:
        store: InMemoryVectorStore instance.
        query: The prompt/query to search for.
        top_k: Number of results to retrieve.
        min_score: Minimum cosine similarity to include.

    Returns:
        Concatenated relevant passages, or empty string if nothing relevant.
    """
    from domains.inference.vector_store import simple_embed

    if store.count_sync() == 0:
        return ""

    query_vec = simple_embed(query, dimension=384)
    results = store.query_sync(query_vec, top_k=top_k)

    relevant = [r.text for r in results if r.score >= min_score]
    if not relevant:
        if results:
            relevant = [results[0].text]

    return "\n\n".join(relevant)


def _build_prompts_from_chunks(
    chunks: List[str],
    prompt_style: str = "continue",
    max_prompts: int = 200,
) -> List[str]:
    """Build prompts from text chunks.

    Args:
        chunks: Text chunks from source.
        prompt_style: How to build prompts:
            - "continue": Use chunk as context, ask to continue
            - "answer": Use chunk as context, ask a question about it
            - "rewrite": Use chunk, ask to rewrite/improve
        max_prompts: Maximum number of prompts to generate.

    Returns:
        List of prompt strings.
    """
    prompts = []
    rng = np.random.default_rng(42)

    question_stems = [
        "Explain this:",
        "Summarize:",
        "What does this mean:",
        "Tell me about:",
        "Describe:",
        "What is:",
        "How does",
        "Can you explain",
        "Write about",
        "Continue this:",
    ]

    for i, chunk in enumerate(chunks[:max_prompts]):
        if prompt_style == "continue":
            sentences = chunk.split(".")
            if len(sentences) > 1:
                prompt = sentences[-2].strip() + "."
            else:
                prompt = chunk[:200]
            prompts.append(prompt)

        elif prompt_style == "answer":
            stem = rng.choice(question_stems)
            first_sent = chunk.split(".")[0].strip()
            if len(first_sent) > 150:
                first_sent = first_sent[:150]
            prompts.append(f"{stem} {first_sent}")

        elif prompt_style == "rewrite":
            prompts.append(f"Rewrite and improve: {chunk[:300]}")

        else:
            prompts.append(chunk[:300])

    return prompts


def _enhance_prompt(store: Any, prompt: str) -> str:
    """Retrieve relevant context from vector store and build an enhanced prompt.

    Args:
        store: InMemoryVectorStore instance.
        prompt: The raw prompt text.

    Returns:
        Enhanced prompt with context prefix, or raw prompt if no context found.
    """
    context = retrieve_context(store, prompt, top_k=3, min_score=0.15)
    if context:
        return f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
    return prompt


def _parse_response(generated: str, enhanced: str, prompt: str) -> str:
    """Extract the teacher's response from raw generated text.

    Strips the enhanced prefix and common response artifacts.

    Args:
        generated: Full text from teacher model.
        enhanced: The enhanced prompt that was fed to the teacher.
        prompt: The original raw prompt.

    Returns:
        Cleaned response string, or empty string if too short.
    """
    if generated.startswith(enhanced):
        response = generated[len(enhanced):].strip()
    elif generated.startswith(prompt):
        response = generated[len(prompt):].strip()
    else:
        response = generated.strip()

    for prefix in ("Answer:", "A:", "Response:"):
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    return response


def _load_teacher_model(
    teacher_model: str,
    max_new_tokens: int,
    temperature: float,
) -> Any:
    """Load a HuggingFace text-generation pipeline.

    Args:
        teacher_model: HuggingFace model name or path.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        HuggingFace text-generation pipeline.

    Raises:
        RuntimeError: If model cannot be loaded.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch

        device = -1
        if torch.cuda.is_available():
            device = 0

        logger.info("Loading teacher model: %s (device=%s)", teacher_model, device)
        return hf_pipeline(
            "text-generation",
            model=teacher_model,
            device=device,
            torch_dtype=torch.float32,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=50256,
            truncation=True,
        )
    except Exception as e:
        logger.error("Failed to load teacher model %s: %s", teacher_model, e)
        raise RuntimeError(f"Cannot load teacher model '{teacher_model}': {e}") from e


def _generate_batch_pipeline(
    teacher,
    enhanced_prompts: List[str],
    batch_prompts: List[str],
    batch_size: int = 8,
) -> List[Dict[str, str]]:
    """Generate pairs from a batch of prompts using a pipeline.

    Args:
        teacher: HuggingFace text-generation pipeline.
        enhanced_prompts: Context-enhanced prompts.
        batch_prompts: Original raw prompts.
        batch_size: Number of prompts per batch.

    Returns:
        List of {"user_msg", "assistant_msg"} dicts.
    """
    pairs = []
    for i, result in enumerate(teacher(enhanced_prompts)):
        prompt = batch_prompts[i]
        enhanced = enhanced_prompts[i]
        response = _parse_response(result[0]["generated_text"], enhanced, prompt)
        if response and len(response) > 10:
            pairs.append({"user_msg": prompt, "assistant_msg": response})
    return pairs


def _generate_pairs_model(
    model,
    tokenizer,
    enhanced_prompts: List[str],
    batch_prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[Dict[str, str]]:
    """Generate pairs from prompts using an already-loaded model+tokenizer.

    Args:
        model: HuggingFace model instance.
        tokenizer: HuggingFace tokenizer instance.
        enhanced_prompts: Context-enhanced prompts.
        batch_prompts: Original raw prompts.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        List of {"user_msg", "assistant_msg"} dicts.
    """
    import torch

    pairs = []
    for enhanced, prompt in zip(enhanced_prompts, batch_prompts):
        try:
            inputs = tokenizer(enhanced, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.01,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id or 0,
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = _parse_response(generated, enhanced, prompt)
            if response and len(response) > 10:
                pairs.append({"user_msg": prompt, "assistant_msg": response})
        except Exception as e:
            logger.warning("Generation failed for prompt: %s", e)
            continue
    return pairs


def generate_teacher_pairs(
    source_text: str,
    teacher_model: str = "gpt2",
    model=None,
    tokenizer=None,
    num_pairs: int = 200,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    prompt_style: str = "continue",
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_batch: Optional[Callable[[List[Dict[str, str]]], None]] = None,
    cancel_event=None,
) -> List[Dict[str, str]]:
    """Generate training pairs using a teacher model with vector-store context.

    The dataset is loaded into a vector store. For each prompt, relevant
    context is retrieved and fed to the teacher alongside the prompt.

    If `model` and `tokenizer` are provided, they are used directly
    (avoids double-loading). Otherwise a pipeline is loaded from
    `teacher_model` name.

    When `on_batch` is provided, it fires after each batch of pairs is
    generated — enabling streaming pipelines where the consumer trains
    on pairs as they arrive rather than waiting for all pairs.

    Args:
        source_text: Raw training text to use as teacher context.
        teacher_model: HuggingFace model name or path (ignored if model given).
        model: HuggingFace model instance (optional, skips loading).
        tokenizer: HuggingFace tokenizer instance (required if model given).
        num_pairs: Maximum number of pairs to generate.
        max_new_tokens: Max tokens per teacher generation.
        temperature: Sampling temperature for teacher.
        prompt_style: How to build prompts from chunks.
        on_progress: Callback(generated_count, total_target).
        on_batch: Callback(batch_pairs) — fires after each batch of pairs.
        cancel_event: threading.Event() to cancel.

    Returns:
        List of {"user_msg": prompt, "assistant_msg": response} dicts.
    """
    use_pipeline = model is None
    model_label = teacher_model if use_pipeline else f"loaded({type(model).__name__})"

    logger.info(
        "Teacher generator: model=%s, source_len=%d, target_pairs=%d",
        model_label, len(source_text), num_pairs,
        extra={"tag": "TRAIN"},
    )

    # Step 1: Load dataset into vector store
    store, chunks = load_dataset_to_store(source_text, chunk_size=400, chunk_overlap=50)
    if not chunks:
        logger.warning("No chunks from source text (len=%d)", len(source_text))
        return []

    # Step 2: Build prompts
    prompts = _build_prompts_from_chunks(
        chunks, prompt_style=prompt_style, max_prompts=num_pairs
    )
    logger.info("Built %d prompts from %d chunks, store has %d entries",
                len(prompts), len(chunks), store.count_sync())

    # Step 3: Load teacher (if not already loaded)
    teacher = None
    if use_pipeline:
        teacher = _load_teacher_model(teacher_model, max_new_tokens, temperature)

    # Step 4: Generate pairs with vector-store context
    pairs: List[Dict[str, str]] = []
    batch_size = 8

    for batch_start in range(0, len(prompts), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.info("Teacher generation cancelled")
            break

        batch_prompts = prompts[batch_start:batch_start + batch_size]
        enhanced_prompts = [_enhance_prompt(store, p) for p in batch_prompts]

        try:
            if use_pipeline:
                batch_pairs = _generate_batch_pipeline(
                    teacher, enhanced_prompts, batch_prompts, batch_size
                )
            else:
                batch_pairs = _generate_pairs_model(
                    model, tokenizer, enhanced_prompts, batch_prompts,
                    max_new_tokens, temperature
                )
            pairs.extend(batch_pairs)

            # Fire streaming callback so consumer can train on this batch
            if on_batch and batch_pairs:
                on_batch(batch_pairs)

        except Exception as e:
            logger.warning("Batch generation failed at %d: %s", batch_start, e)
            continue

        generated_so_far = min(batch_start + batch_size, len(prompts))
        if on_progress:
            on_progress(len(pairs), num_pairs)

        logger.info(
            "Teacher progress: %d/%d pairs from %d/%d prompts (store=%d entries)",
            len(pairs), num_pairs, generated_so_far, len(prompts), store.count_sync(),
        )

        if len(pairs) >= num_pairs:
            break

    pairs = pairs[:num_pairs]

    logger.info(
        "Teacher generation complete: %d pairs from %d prompts (store=%d entries)",
        len(pairs), len(prompts), store.count_sync(),
    )

    return pairs


__all__ = [
    "generate_teacher_pairs",
    "load_dataset_to_store",
    "retrieve_context",
    "_chunk_text",
    "_build_prompts_from_chunks",
    "_enhance_prompt",
    "_parse_response",
]
