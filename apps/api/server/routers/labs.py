"""
Labs Router — experimental endpoints for the Labs playground.

All endpoints here use the auto-train state's model + tokenizer
so the "train → chat" flow works end-to-end in the Playground.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/labs", tags=["labs"])

# In-memory session engine for multi-turn chat
_session_engine: Optional[object] = None
_session_messages: list = []


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    reset: bool = False
    user_id: Optional[str] = None
    user_id_enabled: bool = True


class LoadAndChatRequest(BaseModel):
    checkpoint: str
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8


class ContinueTrainingRequest(BaseModel):
    texts: list[str]
    epochs: int = 5
    learning_rate: float = 0.001


def _get_soul_engine():
    """Create a SloEngine from the current auto-train state."""
    try:
        from routers.auto_train import state as at_state
        from domains.core.soul import SloEngine
        if at_state.student_net is None:
            raise HTTPException(status_code=400, detail="No trained model. Run Playground → Train first.")
        engine = SloEngine(device="cpu")
        engine._model = at_state.student_net
        engine._tokenizer = at_state.student_tokenizer
        if at_state.student_tokenizer and hasattr(at_state.student_tokenizer, "itos"):
            engine._itos = at_state.student_tokenizer.itos
            engine._stoi = at_state.student_tokenizer.stoi
        return engine
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def labs_chat(req: ChatRequest):
    """Chat with the auto-trained model using SloEngine.

    Maintains conversation history for multi-turn context.
    Send ``reset=True`` to clear the conversation.
    """
    global _session_engine, _session_messages

    if req.reset:
        _session_messages = []
        _session_engine = None

    if _session_engine is None:
        _session_engine = _get_soul_engine()

    engine = _session_engine

    # Build prompt with conversation history
    context = "\n".join(f"{m['role']}: {m['content']}" for m in _session_messages[-6:])
    full_prompt = f"{context}\nuser: {req.prompt}\nassistant:" if context else req.prompt

    effective_user_id = req.user_id if req.user_id_enabled else None
    text = engine.generate(
        full_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        include_reasoning=False,
        user_id=effective_user_id,
    )
    for prefix in ["[SOUL_REASONING]", "[REASONING_CHAIN]", "System:", "You are"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    _session_messages.append({"role": "user", "content": req.prompt})
    _session_messages.append({"role": "assistant", "content": text})

    try:
        from domains.learner.entity_extractor import extract_and_store
        extract_and_store(req.prompt, text)
    except Exception:
        pass

    return {"response": text}


@router.post("/chat/stream")
async def labs_chat_stream(req: ChatRequest):
    """Stream chat with the auto-trained model."""
    global _session_engine, _session_messages

    if req.reset:
        _session_messages = []
        _session_engine = None

    if _session_engine is None:
        _session_engine = _get_soul_engine()

    engine = _session_engine

    async def stream():
        from domains.api.sse_envelope import sse_token, sse_complete, sse_error
        try:
            context = "\n".join(f"{m['role']}: {m['content']}" for m in _session_messages[-6:])
            full_prompt = f"{context}\nuser: {req.prompt}\nassistant:" if context else req.prompt

            effective_user_id = req.user_id if req.user_id_enabled else None
            text = engine.generate(
                full_prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                include_reasoning=False,
                user_id=effective_user_id,
            )
            for prefix in ["[SOUL_REASONING]", "[REASONING_CHAIN]", "System:", "You are"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()

            _session_messages.append({"role": "user", "content": req.prompt})
            _session_messages.append({"role": "assistant", "content": text})

            try:
                from domains.learner.entity_extractor import extract_and_store
                extract_and_store(req.prompt, text)
            except Exception:
                pass

            for char in text:
                yield sse_token("labs", char)
                import asyncio
                await asyncio.sleep(0.01)
            yield sse_complete("labs", {}, message="")
        except Exception as e:
            yield sse_error("labs", "STREAMING", str(e))

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/model-info")
async def labs_model_info():
    """Get architecture info for the currently trained model."""
    try:
        from routers.auto_train import state as at_state
        if at_state.student_net is None:
            return {"loaded": False}
        net = at_state.student_net
        tok = at_state.student_tokenizer
        layer_info = []
        for l in net.layers:
            info = {"type": type(l).__name__}
            if hasattr(l, 'weight') and hasattr(l.weight, 'data'):
                info["shape"] = [int(s) for s in l.weight.data.shape]
            if hasattr(l, 'bias') and l.bias is not None and hasattr(l.bias, 'data'):
                info["bias_shape"] = [int(s) for s in l.bias.data.shape]
            layer_info.append(info)
        return {
            "loaded": True,
            "num_parameters": int(net.num_parameters()),
            "layers": layer_info,
            "vocab_size": int(tok.vocab_size) if hasattr(tok, 'vocab_size') else 0,
            "merges": len(tok.merges) if hasattr(tok, 'merges') else 0,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/embeddings")
async def labs_embeddings():
    """Get the embedding weight matrix for heatmap visualization.

    Returns the SloEmbedding weights as a 2D array, normalized
    for display, along with vocab tokens if available.
    """
    try:
        from routers.auto_train import state as at_state
        if at_state.student_net is None:
            return {"error": "No model loaded"}
        net = at_state.student_net
        tok = at_state.student_tokenizer

        # Find the embedding layer
        from domains.training.slonet import SloEmbedding
        embed_layer = None
        for l in net.layers:
            if isinstance(l, SloEmbedding):
                embed_layer = l
                break
        if embed_layer is None:
            return {"error": "No embedding layer found"}

        weights = embed_layer.weight.data
        # Normalize each row to [0, 1] for display
        w_min = weights.min()
        w_max = weights.max()
        normalized = ((weights - w_min) / (w_max - w_min + 1e-10)).tolist()

        # Get sample tokens (up to first 200 vocab entries)
        sample_tokens = []
        if tok and hasattr(tok, 'itos'):
            stoi_rev = {v: k for k, v in tok.stoi.items()} if hasattr(tok, 'stoi') else {}
            for i in range(min(200, len(normalized))):
                token = stoi_rev.get(i, tok.itos.get(str(i)) if hasattr(tok, 'itos') else "")
                sample_tokens.append(token.replace('</w>', '·') if token else f"<{i}>")

        return {
            "weights": normalized,
            "shape": list(weights.shape),
            "sample_tokens": sample_tokens,
            "vocab_size": int(weights.shape[0]),
            "embed_dim": int(weights.shape[1]),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/embedding-map")
async def labs_embedding_map():
    """Get 2D PCA projection of the embedding space for scatter-plot visualization.

    Runs PCA (via SVD) on the embedding weight matrix to project
    each token's embedding vector onto 2 dimensions, showing how
    the model clusters semantically similar tokens.
    """
    try:
        from routers.auto_train import state as at_state
        if at_state.student_net is None:
            return {"error": "No model loaded"}
        net = at_state.student_net
        tok = at_state.student_tokenizer

        from domains.training.slonet import SloEmbedding
        embed_layer = None
        for l in net.layers:
            if isinstance(l, SloEmbedding):
                embed_layer = l
                break
        if embed_layer is None:
            return {"error": "No embedding layer found"}

        import numpy as np
        weights = embed_layer.weight.data
        n_tokens = min(weights.shape[0], 500)

        # Center the data
        X = weights[:n_tokens] - weights[:n_tokens].mean(axis=0)
        # SVD-based PCA
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        proj = U[:, :2] * S[:2]

        # Normalize for display
        proj_min = proj.min(axis=0)
        proj_max = proj.max(axis=0)
        normalized = ((proj - proj_min) / (proj_max - proj_min + 1e-10) * 2 - 1).tolist()

        # Token labels
        sample_tokens = []
        if tok and hasattr(tok, 'stoi'):
            stoi_rev = {v: k for k, v in tok.stoi.items()} if hasattr(tok, 'stoi') else {}
            for i in range(n_tokens):
                token = stoi_rev.get(i, tok.itos.get(str(i)) if hasattr(tok, 'itos') else "")
                sample_tokens.append(token.replace('</w>', '·') if token else f"<{i}>")

        return {
            "points": normalized,
            "tokens": sample_tokens,
            "count": n_tokens,
        }
    except Exception as e:
        return {"error": str(e)}


class TokenProbRequest(BaseModel):
    text: str


@router.post("/token-probs")
async def labs_token_probs(req: TokenProbRequest):
    """Get per-token probability scores for a piece of text.

    Runs the model's forward pass on the text and returns
    the probability assigned to each token given its context.
    High probability = model is confident; low = surprised.
    """
    try:
        from routers.auto_train import state as at_state
        if at_state.student_net is None:
            return {"error": "No model loaded"}
        net = at_state.student_net
        tok = at_state.student_tokenizer
        if tok is None or not hasattr(tok, 'encode'):
            return {"error": "No tokenizer available"}

        from domains.training.slonet import SloLSTM, tensor

        input_ids = tok.encode(req.text[:256])
        if len(input_ids) < 2:
            return {"error": "Text too short"}

        lstm_layers = [l for l in net.layers if isinstance(l, SloLSTM)]
        if not lstm_layers:
            return {"error": "No LSTM layer found"}
        lstm_layer = lstm_layers[0]

        tokens_out = []
        seq_len = len(input_ids)

        # Get token texts
        token_texts = []
        for i in range(seq_len):
            t = tok.decode([input_ids[i]]) if hasattr(tok, 'decode') else f"<{input_ids[i]}>"
            token_texts.append(t.replace('</w>', '·') if t else f"<{input_ids[i]}>")

        # Score each position using the model's prediction from previous context
        for i in range(seq_len - 1):
            context = input_ids[: i + 1]
            context_arr = np.array([context[-64:]], dtype=np.int64)
            in_t = tensor(context_arr, requires_grad=False)
            h = lstm_layer.init_hidden()
            logits_t, _ = lstm_layer.forward(in_t, h)
            logits = logits_t.data[0, -1]
            # Softmax
            logits = np.where(np.isfinite(logits), logits, -1e9)
            ps = np.exp(logits - logits.max())
            ps = ps / (ps.sum() + 1e-10)
            actual_id = input_ids[i + 1]
            prob = float(ps[actual_id]) if actual_id < len(ps) else 0.0

            tokens_out.append({
                "id": actual_id,
                "text": token_texts[i + 1],
                "prob": round(prob, 6),
                "log_prob": round(float(np.log(prob + 1e-10)), 4),
            })

        # Perplexity = exp(avg negative log likelihood)
        log_probs = [t["log_prob"] for t in tokens_out]
        avg_nll = -sum(log_probs) / len(log_probs) if log_probs else 0
        perplexity = round(float(np.exp(avg_nll)), 2)

        return {
            "tokens": tokens_out,
            "perplexity": perplexity,
            "count": len(tokens_out),
            "text_length": len(req.text),
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/import")
async def labs_import(file: UploadFile = File(...)):
    """Import a .soul checkpoint file via multipart upload.

    Saves the uploaded .soul to the auto-training checkpoints directory
    and loads it into the current model.
    """
    try:
        from routers.auto_train import state as at_state, CHECKPOINTS_DIR
        from domains.training.slonet import import_from_sou

        data = await file.read()
        if len(data) < 4:
            return {"error": "File too small or empty"}

        name = file.filename or f"imported_{int(__import__('time').time())}.soul"
        if not name.endswith(".soul"):
            name += ".soul"

        dest = CHECKPOINTS_DIR / name
        dest.write_bytes(data)
        logger = logging.getLogger("labs")
        logger.info(f"Imported {name} ({len(data)} bytes)")

        # Load into model
        loaded = import_from_sou(str(dest))
        at_state.student_net = loaded
        at_state.config["soul_name"] = loaded.soul_name
        if hasattr(loaded, 'soul_traits'):
            at_state.config["traits"] = loaded.soul_traits

        return {
            "success": True,
            "filename": name,
            "size_bytes": len(data),
            "soul": loaded.soul_name,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Agent endpoints ─────────────────────────────────────────────────────

_agent_engine: Optional[object] = None


def _get_agent_engine():
    """Create AgentEngine wrapping the current auto-train model."""
    global _agent_engine
    if _agent_engine is None:
        from domains.agent import AgentEngine, Tool
        engine = _get_soul_engine()
        import datetime as _dt
        tools = [
            Tool(name="get_time", description="Get the current date and time", fn=lambda: _dt.datetime.now().isoformat()),
            Tool(name="echo", description="Echo back the input text", fn=lambda text: text, parameters={"type": "object", "properties": {"text": {"type": "string"}}}),
            Tool(name="count_words", description="Count words in a text string", fn=lambda text: str(len(text.split())), parameters={"type": "object", "properties": {"text": {"type": "string"}}}),
            Tool(name="get_model_info", description="Get the current AI model's status, parameters, and vocabulary size", fn=lambda: json.dumps(_get_soul_engine().status() if hasattr(_get_soul_engine(), 'status') else {"status": "unknown"})),
        ]
        _agent_engine = AgentEngine(engine, tools=tools)
    return _agent_engine


class AgentRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7


@router.post("/agent/run")
async def agent_run(req: AgentRequest):
    """Run the agent with tool calling on the current model."""
    try:
        agent = _get_agent_engine()
        result = agent.run(req.prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        return {
            "response": result.response,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "error": result.error,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/agent/tools")
async def agent_tools():
    """List available tools for the agent."""
    try:
        agent = _get_agent_engine()
        return {"tools": agent.list_tools()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/agent/reset")
async def agent_reset():
    """Reset agent session memory."""
    global _agent_engine
    _agent_engine = None
    return {"status": "reset"}


@router.post("/continue-training")
async def labs_continue_training(req: ContinueTrainingRequest):
    """Continue training the current model with new text."""
    try:
        from routers.auto_train import state as at_state
        from domains.training.slonet import SloAdam, cross_entropy, tensor

        if at_state.student_net is None:
            return {"error": "No model loaded. Train one first."}

        net = at_state.student_net
        tok = at_state.student_tokenizer
        if tok is None or not hasattr(tok, 'encode'):
            return {"error": "No tokenizer available."}

        optimizer = SloAdam(lr=req.learning_rate)
        total_loss = 0.0
        step = 0
        loss_history = []

        for epoch in range(req.epochs):
            for text in req.texts:
                input_ids = tok.encode(text[:128])
                if len(input_ids) < 2:
                    continue
                seq_len = min(len(input_ids) - 1, 32)
                if seq_len < 1:
                    continue
                chunk_size = 8
                for i in range(0, seq_len, chunk_size):
                    x_chunk = input_ids[i:i+chunk_size]
                    y_chunk = input_ids[i+1:i+chunk_size+1]
                    while len(x_chunk) < chunk_size:
                        x_chunk.append(tok.pad_id)
                    while len(y_chunk) < chunk_size:
                        y_chunk.append(tok.pad_id)
                    x = tensor([[x_chunk]], requires_grad=True)
                    y = tensor([[y_chunk]])
                    lstm = net.layers[1]
                    hidden = lstm.init_hidden()
                    logits, _ = lstm.forward(x, hidden)
                    loss = cross_entropy(logits, y.reshape(-1))
                    loss.backward()
                    optimizer.step(net.parameters())
                    step += 1
                    total_loss += loss.data[()]
                    loss_history.append(loss.data[()])

        avg_loss = total_loss / max(step, 1)
        return {
            "success": True,
            "steps": step,
            "loss": round(avg_loss, 6),
            "epochs": req.epochs,
            "texts_used": len(req.texts),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Feedback Training ──────────────────────────────────────────────────


class TrainOnFeedbackRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.001
    include_downvoted: bool = False


@router.post("/train-on-feedback")
async def labs_train_on_feedback(req: TrainOnFeedbackRequest):
    """Train the current model on thumbs-up feedback data.

    Queries the feedback database for rated conversations,
    extracts prompt+response pairs, and continues training
    on the positive examples.
    """
    try:
        from routers.auto_train import state as at_state
        from domains.training.slonet import SloAdam, cross_entropy, tensor
        from domains.feedback.training import FeedbackTrainer

        if at_state.student_net is None:
            return {"error": "No model loaded. Train one first."}

        net = at_state.student_net
        tok = at_state.student_tokenizer
        if tok is None or not hasattr(tok, 'encode'):
            return {"error": "No tokenizer available."}

        # Get training data from feedback DB
        trainer = FeedbackTrainer()
        min_quality = 0.0 if req.include_downvoted else 0.5
        sft_data = trainer.prepare_sft_data(min_quality=min_quality)

        if not sft_data:
            return {"error": "No feedback training data available. Chat and give thumbs up first."}

        # Format as training texts
        texts = []
        for item in sft_data:
            prompt = (item.get("prompt") or "").strip()
            response = (item.get("response") or "").strip()
            if prompt and response:
                texts.append(f"user: {prompt}\nassistant: {response}")

        if not texts:
            return {"error": "No valid prompt/response pairs found."}

        # Train on feedback texts
        optimizer = SloAdam(lr=req.learning_rate)
        total_loss = 0.0
        step = 0
        loss_history = []

        for epoch in range(req.epochs):
            for text in texts:
                input_ids = tok.encode(text[:256])
                if len(input_ids) < 2:
                    continue
                seq_len = min(len(input_ids) - 1, 48)
                if seq_len < 1:
                    continue
                chunk_size = 16
                for i in range(0, seq_len, chunk_size):
                    x_chunk = input_ids[i:i+chunk_size]
                    y_chunk = input_ids[i+1:i+chunk_size+1]
                    while len(x_chunk) < chunk_size:
                        x_chunk.append(tok.pad_id)
                    while len(y_chunk) < chunk_size:
                        y_chunk.append(tok.pad_id)
                    x = tensor([[x_chunk]], requires_grad=True)
                    y = tensor([[y_chunk]])
                    lstm = net.layers[1]
                    hidden = lstm.init_hidden()
                    logits, _ = lstm.forward(x, hidden)
                    loss = cross_entropy(logits, y.reshape(-1))
                    loss.backward()
                    optimizer.step(net.parameters())
                    step += 1
                    total_loss += loss.data[()]
                    loss_history.append(loss.data[()])

        avg_loss = total_loss / max(step, 1)
        return {
            "success": True,
            "steps": step,
            "loss": round(avg_loss, 6),
            "epochs": req.epochs,
            "examples_used": len(texts),
            "total_feedback_pairs": len(sft_data),
            "thumbs_up_only": not req.include_downvoted,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Shell / Terminal ───────────────────────────────────────────────────

import os, shlex, subprocess

_shell_cwd: str = os.getcwd()
_blocked_prefixes = [
    "rm -rf /", "rm -rf ~", "sudo ", "chmod 777 ", "dd if=", "mkfs.",
    ":(){ :|:& };:", "> /dev/sda", "wget ", "curl -o /", "mv /", "reboot",
    "shutdown", "halt", "poweroff", "init ", "kill -9 1",
]


class ExecRequest(BaseModel):
    command: str
    cwd: Optional[str] = None


@router.post("/exec")
async def labs_exec(req: ExecRequest):
    """Execute a shell command in the Labs terminal.

    Runs the command via subprocess with timeout and security checks.
    Tracks working directory across commands (use ``cd`` to navigate).
    """
    global _shell_cwd

    cmd = req.command.strip()
    if not cmd:
        return {"error": "empty command"}

    # Security check
    cmd_lower = cmd.lower()
    for prefix in _blocked_prefixes:
        if cmd_lower.startswith(prefix):
            return {"error": f"Command blocked for safety: {prefix}..."}

    # Handle cd specially
    if cmd.startswith("cd "):
        target = cmd[3:].strip()
        if not target:
            target = os.path.expanduser("~")
        try:
            new_cwd = os.path.abspath(os.path.join(_shell_cwd, target))
            if not os.path.isdir(new_cwd):
                return {"error": f"cd: {target}: No such directory", "cwd": _shell_cwd}
            _shell_cwd = new_cwd
            return {"stdout": "", "stderr": "", "returncode": 0, "cwd": _shell_cwd}
        except Exception as e:
            return {"error": f"cd: {str(e)}", "cwd": _shell_cwd}

    try:
        cmd_list = shlex.split(cmd)
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=req.cwd or _shell_cwd,
        )
        # Update cwd on success
        if req.cwd:
            _shell_cwd = req.cwd
        elif result.returncode == 0:
            pass  # keep current cwd
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "cwd": _shell_cwd,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (15s limit)", "cwd": _shell_cwd}
    except Exception as e:
        return {"error": str(e), "cwd": _shell_cwd}


@router.post("/exec/reset")
async def labs_exec_reset():
    """Reset the shell working directory to the project root."""
    global _shell_cwd
    _shell_cwd = os.getcwd()
    return {"cwd": _shell_cwd}


@router.get("/exec/cwd")
async def labs_exec_cwd():
    """Get current shell working directory."""
    global _shell_cwd
    return {"cwd": _shell_cwd}


class TopKRequest(BaseModel):
    prompt: str
    max_tokens: int = 20
    temperature: float = 0.8
    topn: int = 5


@router.post("/chat/topk")
async def labs_chat_topk(req: TopKRequest):
    """Chat with the model and return top-k token predictions per step.

    Shows what the model considered at each generation step.
    """
    try:
        engine = _get_soul_engine()
        steps = engine.generate_with_topk(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            topn=req.topn,
        )
        return {"steps": steps, "prompt": req.prompt}
    except Exception as e:
        return {"error": str(e)}


@router.get("/feedback-stats")
async def labs_feedback_stats():
    """Get feedback statistics for the Labs UI.

    Shows how much training data is available from feedback.
    """
    try:
        from domains.feedback.database import get_feedback_db
        db = get_feedback_db()
        stats = db.get_stats()

        from domains.feedback.training import FeedbackTrainer
        trainer = FeedbackTrainer()
        train_stats = trainer.get_training_stats()

        return {
            "total_feedback": stats.get("feedback_total", 0),
            "thumbs_up": stats.get("thumbs_up", 0),
            "thumbs_down": stats.get("thumbs_down", 0),
            "messages": stats.get("messages", 0),
            "conversations": stats.get("conversations", 0),
            "available_dpo_pairs": train_stats.get("available_dpo_pairs", 0),
            "available_sft_examples": train_stats.get("available_sft_examples", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/health")
async def labs_health():
    """Get model health status including perplexity trend and drift detection.

    Returns current PPL, best/worst/average, trend points for charting,
    and drift detection results.
    """
    try:
        from domains.feedback.model_health import get_health_monitor
        monitor = get_health_monitor()
        return monitor.get_stats()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/health/benchmark")
async def labs_health_benchmark():
    """Manually trigger a benchmark run."""
    try:
        from routers.auto_train import state as at_state
        from domains.feedback.model_health import get_health_monitor

        monitor = get_health_monitor()
        net = at_state.student_net
        tok = at_state.student_tokenizer
        if net is None or tok is None:
            return {"error": "No trained model loaded"}

        monitor.set_model(net, tok)
        snapshot = monitor.run_benchmark()
        if snapshot is None:
            return {"error": "Benchmark failed"}
        return {
            "perplexity": round(snapshot.perplexity, 4),
            "loss": round(snapshot.loss, 4),
            "num_sentences": snapshot.num_sentences,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/health-status")
async def labs_health_status():
    """Get combined health, workflow, and adapter status.

    Returns:
        - ``model``: model health monitor stats (PPL, drift, benchmarks)
        - ``workflow``: scheduled task status (last runs, rollback time,
          aggregated/performed stats)
        - ``adapters``: per‑user adapter store summary (total users, size,
          quality count)
        - ``available``: overall health indicator
    """
    try:
        from domains.feedback.model_health import get_health_monitor
        from domains.feedback.workflow import get_feedback_workflow
        from domains.feedback.per_user_lora import get_per_user_lora

        monitor = get_health_monitor()
        wf = get_feedback_workflow()
        lora_store = get_per_user_lora()

        health = monitor.get_stats()
        workflow = wf.get_status()
        adapters = lora_store.get_stats()

        return {
            "available": True,
            "model": health,
            "workflow": workflow,
            "adapters": adapters,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


class AddKnowledgeRequest(BaseModel):
    content: str
    category: str = "general"
    tags: list[str] = []


class UpdateKnowledgeRequest(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None


@router.get("/knowledge")
def list_knowledge():
    """List all stored facts in KnowledgeMemory."""
    try:
        from domains.learner.knowledge import get_knowledge_memory
        km = get_knowledge_memory()
        entries = km.list_all()
        return {"items": entries, "count": len(entries)}
    except Exception as e:
        return {"items": [], "count": 0, "error": str(e)}


@router.post("/knowledge")
def add_knowledge(req: AddKnowledgeRequest):
    """Add a new fact to KnowledgeMemory."""
    from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
    km = get_knowledge_memory()
    fact = KnowledgeFact(
        content=req.content,
        topic=req.category or "manual",
        source="labs",
        importance=0.5,
    )
    stored = km.add_fact(fact)
    return {"stored": stored}


@router.get("/knowledge/search")
def search_knowledge(query: str, category: Optional[str] = None):
    """Search stored facts by semantic similarity."""
    from domains.learner.knowledge import get_knowledge_memory
    km = get_knowledge_memory()
    results = km.search(query, top_k=20) if query else []
    if category:
        results = [r for r in results if r.get("topic") == category]
    return {"items": results, "count": len(results)}


@router.delete("/knowledge")
def clear_all_knowledge():
    """Clear all stored facts from KnowledgeMemory."""
    from domains.learner.knowledge import get_knowledge_memory
    km = get_knowledge_memory()
    removed = km.clear_all()
    return {"cleared": True, "removed": removed}
