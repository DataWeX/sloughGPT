"""
Layer 4 device drivers for the AI Networking Processor VM.

These translate VM register values into Python library calls.  Each
device wraps an existing Python object (numpy, SlonetChatProvider,
MultimodalEngine, etc.) and exposes a ``call(method, *args)`` interface
that the VM dispatches to via ``DEV_OPEN`` / ``DEV_CALL`` /
``DEV_CLOSE`` instructions.
"""

from __future__ import annotations

import numpy as np

from .vm import Device, DeviceFault


# ---------------------------------------------------------------------------
# TensorDevice – wraps numpy
# ---------------------------------------------------------------------------

class TensorDevice(Device):
    """
    Tensor operations device — wraps numpy.

    From assembly:
        DEV_OPEN   R0, tensor
        DEV_CALL   R1, R0, matmul, R2, R3      # R1 = R2 @ R3
        DEV_CALL   R1, R0, relu, R2             # R1 = relu(R2)
        DEV_CALL   R1, R0, softmax, R2          # R1 = softmax(R2)
        DEV_CALL   R1, R0, add, R2, R3          # R1 = R2 + R3
        DEV_CALL   R1, R0, forward, R2          # R1 = full MLP forward pass
    """

    def __init__(self, weights=None):
        self._weights = dict(weights) if weights else {}
        self._ops = {
            "matmul": self._matmul,
            "relu": self._relu,
            "softmax": self._softmax,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "add": self._add,
            "mul": self._mul,
            "sub": self._sub,
            "neg": self._neg,
            "abs": self._abs,
            "sum": self._sum,
            "mean": self._mean,
            "max": self._max,
            "argmax": self._argmax,
            "norm": self._norm,
            "load": self._load_weight,
            "store": self._store_weight,
            "shape": self._shape,
            "zeros": self._zeros,
            "randn": self._randn,
            "forward": self._forward,
            "info": self.info,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"TensorDevice: unknown op: {method}")
        return fn(*args)

    def info(self):
        return {
            "type": "tensor",
            "ops": list(self._ops.keys()),
            "weight_names": list(self._weights.keys()),
        }

    def _to_arr(self, v):
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (int, float)):
            return np.float64(v)
        if isinstance(v, list):
            return np.array(v, dtype=np.float64)
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return np.array(parsed, dtype=np.float64)
            except (ValueError, TypeError, MemoryError):
                pass
        return np.array(v, dtype=np.float64)

    def _matmul(self, a, b):
        return self._to_arr(a) @ self._to_arr(b)

    def _relu(self, a):
        return np.maximum(0, self._to_arr(a))

    def _softmax(self, a):
        x = self._to_arr(a)
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)

    def _sigmoid(self, a):
        x = self._to_arr(a)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, a):
        return np.tanh(self._to_arr(a))

    def _add(self, a, b):
        return self._to_arr(a) + self._to_arr(b)

    def _mul(self, a, b):
        return self._to_arr(a) * self._to_arr(b)

    def _sub(self, a, b):
        return self._to_arr(a) - self._to_arr(b)

    def _neg(self, a):
        return -self._to_arr(a)

    def _abs(self, a):
        return np.abs(self._to_arr(a))

    def _sum(self, a):
        return float(np.sum(self._to_arr(a)))

    def _mean(self, a):
        return float(np.mean(self._to_arr(a)))

    def _max(self, a):
        return float(np.max(self._to_arr(a)))

    def _argmax(self, a):
        return int(np.argmax(self._to_arr(a)))

    def _norm(self, a):
        return float(np.linalg.norm(self._to_arr(a)))

    def _load_weight(self, name):
        name = str(name)
        if name not in self._weights:
            raise DeviceFault(f"TensorDevice: no weight: {name}")
        return self._weights[name]

    def _store_weight(self, name, value):
        self._weights[str(name)] = self._to_arr(value)

    def _shape(self, a):
        return list(self._to_arr(a).shape)

    def _zeros(self, rows, cols):
        r = int(rows) if isinstance(rows, (int, float)) else 1
        c = int(cols) if isinstance(cols, (int, float)) else 1
        return np.zeros((r, c), dtype=np.float64)

    def _randn(self, rows, cols):
        r = int(rows) if isinstance(rows, (int, float)) else 1
        c = int(cols) if isinstance(cols, (int, float)) else 1
        return np.random.randn(r, c)

    def _forward(self, input_vec):
        x = self._to_arr(input_vec).ravel()
        w1 = self._weights.get("w1")
        b1 = self._weights.get("b1")
        w2 = self._weights.get("w2")
        b2 = self._weights.get("b2")
        if w1 is None or w2 is None:
            raise DeviceFault("TensorDevice: forward requires w1, w2 weights")
        h = w1 @ x
        if b1 is not None:
            h = h + b1
        h = np.maximum(0, h)
        out = w2 @ h
        if b2 is not None:
            out = out + b2
        shifted = out - np.max(out)
        exp_out = np.exp(shifted)
        probs = exp_out / np.sum(exp_out)
        return probs


# ---------------------------------------------------------------------------
# PythonExecDevice – execute arbitrary Python from assembly
# ---------------------------------------------------------------------------

class PythonExecDevice(Device):
    """
    Execute arbitrary Python from assembly.

    From assembly:
        DEV_OPEN   R0, python
        DEV_CALL   R1, R0, eval, "2 + 2"       # R1 = 4
        DEV_CALL   R1, R0, call, len, [1,2,3]   # R1 = 3
        DEV_CALL   R1, R0, import, numpy         # R1 = <module>
        DEV_CALL   R1, R0, exec, "x = 5"         # R1 = None (side effect)
    """

    _SAFE_BUILTINS = {
        "len": len, "range": range, "int": int, "float": float,
        "str": str, "bool": bool, "list": list, "dict": dict,
        "print": print, "min": min, "max": max, "sum": sum,
        "abs": abs, "round": round, "sorted": sorted, "reversed": reversed,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "isinstance": isinstance, "type": type, "hasattr": hasattr,
        "getattr": getattr, "setattr": setattr,
    }

    def __init__(self):
        self._scope = {"np": np, "numpy": np}

    def call(self, method, *args):
        if method == "eval":
            return self._py_eval(*args)
        elif method == "call":
            return self._py_call(*args)
        elif method == "import":
            return self._py_import(*args)
        elif method == "exec":
            return self._py_exec(*args)
        elif method == "set":
            return self._py_set(*args)
        elif method == "get":
            return self._py_get(*args)
        elif method == "scope":
            return dict(self._scope)
        else:
            raise DeviceFault(f"PythonExecDevice: unknown op: {method}")

    def info(self):
        return {
            "type": "python_exec",
            "ops": ["eval", "call", "import", "exec", "set", "get", "scope"],
        }

    def _py_eval(self, expr):
        return eval(str(expr), {"__builtins__": self._SAFE_BUILTINS}, self._scope)

    def _py_call(self, func_name, *args):
        name = str(func_name)
        if name in self._scope:
            fn = self._scope[name]
        elif name in self._SAFE_BUILTINS:
            fn = self._SAFE_BUILTINS[name]
        else:
            raise DeviceFault(f"PythonExecDevice: unknown callable: {name}")
        return fn(*args)

    def _py_import(self, module_name):
        import importlib
        mod = importlib.import_module(str(module_name))
        self._scope[str(module_name)] = mod
        return mod

    def _py_exec(self, code):
        exec(str(code), {"__builtins__": {}}, self._scope)
        return None

    def _py_set(self, name, value):
        self._scope[str(name)] = value

    def _py_get(self, name):
        name = str(name)
        if name not in self._scope:
            raise DeviceFault(f"PythonExecDevice: undefined: {name}")
        return self._scope[name]


# ---------------------------------------------------------------------------
# SlonetDevice – text inference via SlonetChatProvider
# ---------------------------------------------------------------------------

class SlonetDevice(Device):
    """
    Text inference device — wraps SlonetChatProvider.

    From assembly:
        DEV_OPEN   R0, slonet
        DEV_CALL   R1, R0, tokenize, "Hello world"     # R1 = [token_ids]
        DEV_CALL   R2, R0, generate, R1, 50             # R2 = [generated_ids]
        DEV_CALL   R3, R0, detokenize, R2               # R3 = "Hello world..."
        DEV_CALL   R4, R0, forward, R1                   # R4 = logits
        DEV_CALL   R5, R0, info                          # R5 = model info
    """

    def __init__(self, provider):
        self._provider = provider
        self._ops = {
            "tokenize": self._tokenize,
            "detokenize": self._detokenize,
            "generate": self._generate,
            "generate_stream": self._generate_stream,
            "forward": self._forward,
            "info": self._info,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"SlonetDevice: unknown op: {method}")
        return fn(*args)

    def _tokenize(self, text):
        tokens = self._provider._tokenizer.encode(str(text))
        return np.array(tokens, dtype=np.int64)

    def _detokenize(self, token_ids):
        ids = np.asarray(token_ids).ravel().tolist()
        return self._provider._tokenizer.decode(ids)

    def _generate(self, token_ids, max_tokens=50, temperature=1.0,
                  top_k=None, top_p=None, repetition_penalty=1.0):
        input_ids = np.asarray(token_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        result = self._provider._model.generate_numpy(
            input_ids,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            repetition_penalty=float(repetition_penalty),
            eos_token=self._provider._tokenizer.eos_token_id or 0,
        )
        return result[0]

    def _generate_stream(self, token_ids, max_tokens=50, eos_token=0):
        input_ids = np.asarray(token_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        for token_id in self._provider._model.generate_numpy_stream(
            input_ids, max_new_tokens=int(max_tokens), eos_token=int(eos_token),
            temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0,
        ):
            yield token_id

    def _forward(self, token_ids):
        input_ids = np.asarray(token_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        from domains.training.slonet import Tensor as _Tensor
        inp = _Tensor(input_ids, requires_grad=False)
        logits, _ = self._provider._model.forward(inp)
        return logits.data

    def _info(self):
        m = self._provider._model
        return {
            "model_id": self._provider._model_id,
            "vocab_size": m.vocab_size,
            "n_embed": m.n_embed,
            "n_layer": len(m.layers) if hasattr(m, 'layers') else 0,
            "block_size": m.block_size if hasattr(m, 'block_size') else 0,
        }


# ---------------------------------------------------------------------------
# MultimodalDevice – vision + text inference via MultimodalEngine
# ---------------------------------------------------------------------------

class MultimodalDevice(Device):
    """
    Vision + text inference device — wraps MultimodalEngine.

    From assembly:
        DEV_OPEN   R0, multimodal
        DEV_CALL   R1, R0, generate, R_img, 20, 1.0      # R1 = MultimodalOutput
        DEV_CALL   R2, R0, embed, R_img                   # R2 = feature vector
        DEV_CALL   R3, R0, info                            # R3 = model info

    Image is passed as a numpy array (H×W×3, float32 [0,1]).
    """

    def __init__(self, engine):
        self._engine = engine
        self._ops = {
            "generate": self._generate,
            "embed": self._embed,
            "info": self._info,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"MultimodalDevice: unknown op: {method}")
        return fn(*args)

    def _generate(self, image_np=None, max_len=20, temperature=1.0):
        img = np.asarray(image_np, dtype=np.float32) if image_np is not None else None
        if img is not None and img.ndim == 3:
            img = img[np.newaxis, ...]
        result = self._engine.generate(
            image_np=img, max_len=int(max_len), temperature=float(temperature),
        )
        return result.text

    def _embed(self, image_np):
        img = np.asarray(image_np, dtype=np.float32)
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        embed, patches, _ = self._engine._concat_modalities(img, None, None)
        return embed.data if hasattr(embed, 'data') else np.asarray(embed)

    def _info(self):
        return {
            "trained": self._engine._trained,
            "embed_dim": self._engine.vision.embed_dim if hasattr(self._engine.vision, 'embed_dim') else 0,
        }


# ---------------------------------------------------------------------------
# EngineDevice – generic inference via any engine with generate()
# ---------------------------------------------------------------------------

class EngineDevice(Device):
    """
    Generic inference device — wraps any engine with a generate() method.

    From assembly:
        DEV_OPEN   R0, engine
        DEV_CALL   R1, R0, generate, "Hello", 50           # R1 = response text
        DEV_CALL   R2, R0, info                             # R2 = engine info
    """

    def __init__(self, engine, engine_name="engine"):
        self._engine = engine
        self._name = engine_name
        self._ops = {
            "generate": self._generate,
            "info": self._info,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"EngineDevice({self._name}): unknown op: {method}")
        return fn(*args)

    def _generate(self, prompt, max_tokens=50, temperature=1.0, **kwargs):
        return self._engine.generate(
            str(prompt), max_tokens=int(max_tokens), temperature=float(temperature), **kwargs,
        )

    def _info(self):
        return {
            "engine": self._name,
            "type": type(self._engine).__name__,
        }


# ---------------------------------------------------------------------------
# SlonetTrainingDevice – full training device (train/eval/checkpoint)
# ---------------------------------------------------------------------------

class SlonetTrainingDevice(Device):
    """
    Full training device — wraps SloTransformer with train/eval/checkpoint.

    From assembly:
        DEV_OPEN   R7, slonet_train
        DEV_CALL   R1, R7, train, 10, /data/shakespeare.txt, 0.0003, 8, 128
        DEV_CALL   R2, R7, eval, /data/shakespeare.txt
        DEV_CALL   R3, R7, save, /disks/main/checkpoints/model.soul
        DEV_CALL   R4, R7, load, /disks/main/checkpoints/model.soul
        DEV_CALL   R5, R7, config, lr, 0.001
        DEV_CLOSE  R7

    Two modes:
        1. Wrap existing model:  SlonetTrainingDevice(model=loaded_model)
        2. Create from config:   SlonetTrainingDevice(vocab_size=256, n_embed=256, ...)
    """

    def __init__(self, model=None, *, vocab_size=256, n_embed=256, n_layer=6,
                 n_head=8, block_size=128, dropout=0.1):
        if model is not None:
            self._model = model
            self._created_model = False
        else:
            self._model = None
            self._model_config = {
                "vocab_size": int(vocab_size),
                "n_embed": int(n_embed),
                "n_layer": int(n_layer),
                "n_head": int(n_head),
                "block_size": int(block_size),
                "dropout": float(dropout),
            }
            self._created_model = True

        self._train_config = {
            "lr": 3e-4,
            "batch_size": 8,
            "max_seq_len": 128,
            "epochs": 10,
            "max_grad_norm": 1.0,
            "scheduler": "warmup_cosine",
            "warmup_steps": 100,
            "save_interval": 0,
            "checkpoint_dir": "checkpoints",
        }

        self._ops = {
            "train": self._train,
            "eval": self._eval,
            "save": self._save,
            "load": self._load,
            "config": self._config_op,
            "info": self._info,
            "tokenize": self._tokenize,
            "detokenize": self._detokenize,
            "generate": self._generate,
            "forward": self._forward,
        }

    def _ensure_model(self):
        if self._model is None:
            from domains.training.slonet import SloTransformer
            self._model = SloTransformer(**self._model_config)

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"SlonetTrainingDevice: unknown op: {method}")
        return fn(*args)

    def _load_dataset(self, path: str, max_seq_len: int):
        try:
            from domains.shell.file_manager import get_file_manager
            fm = get_file_manager()
            content = fm.read_text(path)
        except ImportError:
            import os
            from pathlib import Path
            expanded = os.path.expanduser(str(path))
            try:
                content = Path(expanded).read_text()
            except (OSError, PermissionError):
                content = None

        if content is None or len(content) == 0:
            return None

        chars = sorted(set(content))
        stoi = {c: i for i, c in enumerate(chars)}
        vocab_size = len(chars)
        token_ids = np.array([stoi[c] for c in content], dtype=np.int64)
        return token_ids, vocab_size

    def _train(self, epochs=10, dataset_path="", lr=3e-4,
               batch_size=8, max_seq_len=128, save_interval=0):
        self._ensure_model()

        from domains.training.slonet import (
            cross_entropy, SloAdam, clip_grad_norm_, Tensor, export_to_sou
        )

        data = self._load_dataset(dataset_path, max_seq_len)
        if data is None:
            return {"error": f"Could not load dataset: {dataset_path}"}

        token_ids, vocab_size = data
        n_tokens = len(token_ids)
        steps_per_epoch = max(1, (n_tokens - max_seq_len) // batch_size)
        total_steps = epochs * steps_per_epoch

        params = self._model.parameters()
        optimizer = SloAdam(lr=lr, max_grad_norm=self._train_config["max_grad_norm"])

        warmup_steps = min(self._train_config["warmup_steps"], max(1, total_steps // 4))
        try:
            from domains.training.slonet import WarmupCosineScheduler
            scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps,
                                              total_steps=total_steps, min_lr=lr * 0.1)
            has_scheduler = True
        except ImportError:
            has_scheduler = False

        global_step = 0
        losses = []

        for epoch in range(epochs):
            indices = np.random.permutation(max(1, n_tokens - max_seq_len))

            for step in range(steps_per_epoch):
                batch_indices = indices[step * batch_size : (step + 1) * batch_size]

                x_batch = np.array([token_ids[i:i + max_seq_len] for i in batch_indices], dtype=np.int64)
                y_batch = np.array([token_ids[i + 1:i + max_seq_len + 1] for i in batch_indices], dtype=np.int64)

                x_tensor = Tensor(x_batch)
                y_tensor = Tensor(y_batch)
                logits, loss = self._model(x_tensor, y_tensor)

                loss.backward()
                clip_grad_norm_(params, max_norm=self._train_config["max_grad_norm"])
                optimizer.step(params)

                if has_scheduler:
                    scheduler.step()

                raw_loss = float(loss.data)
                losses.append(raw_loss)
                global_step += 1

                if save_interval > 0 and global_step % save_interval == 0:
                    import os
                    os.makedirs(self._train_config["checkpoint_dir"], exist_ok=True)
                    ckpt_path = f"{self._train_config['checkpoint_dir']}/step_{global_step}.soul"
                    export_to_sou(self._model, ckpt_path,
                                  metadata={"step": global_step, "loss": raw_loss})

        final_loss = float(np.mean(losses[-100:])) if losses else 0.0
        perplexity = float(np.exp(np.clip(final_loss, -10, 10)))

        import os, time
        os.makedirs(self._train_config["checkpoint_dir"], exist_ok=True)
        ckpt_name = f"train_{int(time.time())}.soul"
        ckpt_path = os.path.join(self._train_config["checkpoint_dir"], ckpt_name)
        export_to_sou(self._model, ckpt_path, metadata={
            "final_loss": final_loss, "perplexity": perplexity,
            "epochs": epochs, "steps": global_step,
            "tokens": global_step * batch_size * max_seq_len,
        })

        return {
            "final_loss": round(final_loss, 4),
            "perplexity": round(perplexity, 4),
            "epochs_completed": epochs,
            "total_tokens": global_step * batch_size * max_seq_len,
            "steps": global_step,
            "saved_path": ckpt_path,
        }

    def _eval(self, dataset_path="", max_seq_len=128, num_batches=50):
        self._ensure_model()

        from domains.training.slonet import Tensor

        data = self._load_dataset(dataset_path, max_seq_len)
        if data is None:
            return {"error": f"Could not load dataset: {dataset_path}"}

        token_ids, vocab_size = data
        n_tokens = len(token_ids)
        batch_size = self._train_config["batch_size"]
        actual_batches = min(num_batches, max(1, (n_tokens - max_seq_len) // batch_size))

        losses = []
        for _ in range(actual_batches):
            idx = np.random.randint(0, max(1, n_tokens - max_seq_len))
            x = token_ids[idx:idx + max_seq_len]
            y = token_ids[idx + 1:idx + max_seq_len + 1]
            x_tensor = Tensor(x.reshape(1, -1))
            y_tensor = Tensor(y.reshape(1, -1))
            logits, loss = self._model(x_tensor, y_tensor)
            losses.append(float(loss.data))

        avg_loss = float(np.mean(losses))
        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(float(np.exp(np.clip(avg_loss, -10, 10))), 4),
            "batches_evaluated": actual_batches,
            "tokens_evaluated": actual_batches * max_seq_len,
        }

    def _save(self, path=""):
        self._ensure_model()
        from domains.training.slonet import export_to_sou
        import os, time
        if not path:
            os.makedirs(self._train_config["checkpoint_dir"], exist_ok=True)
            path = os.path.join(self._train_config["checkpoint_dir"],
                                f"manual_{int(time.time())}.soul")
        export_to_sou(self._model, path)
        return path

    def _load(self, path=""):
        from domains.training.slonet import import_from_sou
        try:
            self._model = import_from_sou(path)
            self._created_model = False
            return f"loaded from {path}"
        except Exception as e:
            return f"load error: {e}"

    def _config_op(self, key="", value=""):
        if not key:
            return dict(self._train_config)
        if key in self._train_config:
            if value != "":
                current = self._train_config[key]
                try:
                    if isinstance(current, int):
                        self._train_config[key] = int(value)
                    elif isinstance(current, float):
                        self._train_config[key] = float(value)
                    else:
                        self._train_config[key] = value
                except (ValueError, TypeError):
                    self._train_config[key] = value
            return {key: self._train_config[key]}
        return {"error": f"unknown config key: {key}"}

    def _tokenize(self, text):
        text = str(text)
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return np.array([stoi.get(c, 0) for c in text], dtype=np.int64)

    def _detokenize(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.flatten().tolist()
        elif isinstance(ids, (int, float)):
            ids = [int(ids)]
        return "".join(chr(max(32, min(126, i))) for i in ids)

    def _generate(self, prompt, max_tokens=50, temperature=1.0):
        self._ensure_model()
        from domains.training.slonet import Tensor
        tokens = self._tokenize(str(prompt))
        if len(tokens) == 0:
            return ""
        x = Tensor(tokens.reshape(1, -1))
        logits, _ = self._model(x)
        last_logits = logits.data[0, -1, :]
        if temperature > 0:
            last_logits = last_logits / temperature
        next_token = int(np.argmax(last_logits))
        return self._detokenize([next_token])

    def _forward(self, input_ids):
        self._ensure_model()
        from domains.training.slonet import Tensor
        if isinstance(input_ids, np.ndarray):
            x = Tensor(input_ids.reshape(1, -1) if input_ids.ndim == 1 else input_ids)
        else:
            x = Tensor(np.array(input_ids, dtype=np.int64).reshape(1, -1))
        logits, _ = self._model(x)
        return logits.data

    def _info(self):
        return {
            "type": "slonet_train",
            "ops": list(self._ops.keys()),
            "model_created": self._created_model,
            "config": dict(self._train_config),
        }


# ---------------------------------------------------------------------------
# NPUVMDevice – VM-level NPU device bridging to kernel NPUDevice
# ---------------------------------------------------------------------------

class NPUVMDevice(Device):
    """
    VM-level NPU device — bridges assembly to kernel NPUDevice.

    From assembly:
        DEV_OPEN   R0, npu
        DEV_CALL   R1, R0, load_model, qwen, hf://qwen2.5-0.5B-Instruct
        DEV_CALL   R2, R0, tokenize, qwen, Hello world
        DEV_CALL   R3, R0, generate, qwen, Hello, 50
        DEV_CALL   R4, R0, detokenize, qwen, R3
        DEV_CALL   R5, R0, embed, qwen, Hello world
        DEV_CALL   R6, R0, forward, qwen, R2
        DEV_CALL   R7, R0, info
        DEV_CALL   R8, R0, unload_model, qwen
        DEV_CLOSE  R0

    Wraps kernel.npu.NPUDevice behind the VM Device protocol.
    """

    def __init__(self, npu_device):
        self._npu = npu_device
        self._ops = {
            "load_model": self._load_model,
            "unload_model": self._unload_model,
            "tokenize": self._tokenize,
            "detokenize": self._detokenize,
            "generate": self._generate,
            "forward": self._forward,
            "embed": self._embed,
            "train_step": self._train_step,
            "profile": self._profile,
            "checkpoint": self._checkpoint,
            "restore": self._restore,
            "list_checkpoints": self._list_checkpoints,
            "delete_checkpoint": self._delete_checkpoint,
            "save_checkpoint": self._save_checkpoint,
            "load_checkpoint": self._load_checkpoint,
            "quantize": self._quantize,
            "dequantize": self._dequantize,
            "clear_cache": self._clear_cache,
            "health": self._health,
            "batch": self._batch,
            "attention_maps": self._attention_maps,
            "compare": self._compare,
            "layers": self._layers,
            "benchmark": self._benchmark,
            "info": self._info,
        }

    def call(self, method, *args, **kwargs):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"NPUVMDevice: unknown op: {method}")
        return fn(*args, **kwargs)

    def _load_model(self, name, source, **kwargs):
        result = self._npu.load_file(str(source), str(name), **kwargs)
        if not result.success:
            raise DeviceFault(f"NPU load_file failed: {result.error}")
        return result.value

    def _unload_model(self, name):
        result = self._npu.unload_model(str(name))
        if not result.success:
            raise DeviceFault(f"NPU unload_model failed: {result.error}")
        return result.value

    def _tokenize(self, name, text):
        result = self._npu.tokenize(str(name), str(text))
        if not result.success:
            raise DeviceFault(f"NPU tokenize failed: {result.error}")
        return result.value.get("token_ids", [])

    def _detokenize(self, name, token_ids):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        result = self._npu.detokenize(str(name), token_ids)
        if not result.success:
            raise DeviceFault(f"NPU detokenize failed: {result.error}")
        return result.value.get("text", "")

    def _generate(self, name, prompt, max_tokens=100, **kwargs):
        result = self._npu.generate(str(name), str(prompt), int(max_tokens), **kwargs)
        if not result.success:
            raise DeviceFault(f"NPU generate failed: {result.error}")
        return result.value.get("text", "")

    def _forward(self, name, input_ids):
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        elif not isinstance(input_ids, list):
            input_ids = list(input_ids)
        result = self._npu.forward(str(name), input_ids)
        if not result.success:
            raise DeviceFault(f"NPU forward failed: {result.error}")
        return result.value.get("logits", None)

    def _embed(self, name, text, layer=-1):
        result = self._npu.embed(str(name), str(text), int(layer))
        if not result.success:
            raise DeviceFault(f"NPU embed failed: {result.error}")
        return result.value.get("embedding", None)

    def _train_step(self, name, input_ids, targets, lr=0.001, **kwargs):
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        if hasattr(targets, 'tolist'):
            targets = targets.tolist()
        result = self._npu.train_step(str(name), input_ids, targets, float(lr), **kwargs)
        if not result.success:
            raise DeviceFault(f"NPU train_step failed: {result.error}")
        return result.value

    def _info(self):
        return self._npu.info()

    def _profile(self, name="", seq_len=512, batch_sizes=None):
        if batch_sizes is not None and isinstance(batch_sizes, str):
            batch_sizes = [int(x) for x in batch_sizes.split(",")]
        return self._npu.profile(str(name), int(seq_len), batch_sizes)

    def _checkpoint(self, name="", checkpoint_name=""):
        result = self._npu.checkpoint(str(name), str(checkpoint_name))
        if not result.success:
            raise DeviceFault(f"NPU checkpoint failed: {result.error}")
        return result.value

    def _restore(self, name="", checkpoint_name=""):
        result = self._npu.restore(str(name), str(checkpoint_name))
        if not result.success:
            raise DeviceFault(f"NPU restore failed: {result.error}")
        return result.value

    def _list_checkpoints(self):
        result = self._npu.list_checkpoints()
        if not result.success:
            raise DeviceFault(f"NPU list_checkpoints failed: {result.error}")
        return result.value

    def _delete_checkpoint(self, checkpoint_name):
        result = self._npu.delete_checkpoint(str(checkpoint_name))
        if not result.success:
            raise DeviceFault(f"NPU delete_checkpoint failed: {result.error}")
        return result.value

    def _save_checkpoint(self, name="", path=""):
        result = self._npu.save_checkpoint(str(name), str(path))
        if not result.success:
            raise DeviceFault(f"NPU save_checkpoint failed: {result.error}")
        return result.value

    def _load_checkpoint(self, name="", path=""):
        result = self._npu.load_checkpoint(str(name), str(path))
        if not result.success:
            raise DeviceFault(f"NPU load_checkpoint failed: {result.error}")
        return result.value

    def _quantize(self, name="", bits=8):
        result = self._npu.quantize(str(name), int(bits))
        if not result.success:
            raise DeviceFault(f"NPU quantize failed: {result.error}")
        return result.value

    def _dequantize(self, name=""):
        result = self._npu.dequantize(str(name))
        if not result.success:
            raise DeviceFault(f"NPU dequantize failed: {result.error}")
        return result.value

    def _clear_cache(self, name=""):
        result = self._npu.clear_cache(str(name))
        if not result.success:
            raise DeviceFault(f"NPU clear_cache failed: {result.error}")
        return result.value

    def _health(self):
        result = self._npu.health()
        if not result.success:
            raise DeviceFault(f"NPU health check failed: {result.error}")
        return result.value

    def _batch(self, name="", prompts=None, max_tokens=50):
        if prompts is None:
            prompts = []
        if isinstance(prompts, str):
            prompts = [p.strip() for p in prompts.split("|")]
        result = self._npu.batch(str(name), prompts, int(max_tokens))
        if not result.success:
            raise DeviceFault(f"NPU batch failed: {result.error}")
        return result.value

    def _attention_maps(self, name="", text="", layer=-1):
        result = self._npu.attention_maps(str(name), str(text), int(layer))
        if not result.success:
            raise DeviceFault(f"NPU attention_maps failed: {result.error}")
        return result.value

    def _compare(self, model_a="", model_b="", prompt="Hello", max_tokens=20):
        result = self._npu.compare(str(model_a), str(model_b), str(prompt), int(max_tokens))
        if not result.success:
            raise DeviceFault(f"NPU compare failed: {result.error}")
        return result.value

    def _layers(self, name="", layer=-1):
        result = self._npu.layers(str(name), int(layer))
        if not result.success:
            raise DeviceFault(f"NPU layers failed: {result.error}")
        return result.value

    def _benchmark(self, name="", prompt_lengths=None, max_tokens=50):
        if isinstance(prompt_lengths, str):
            prompt_lengths = [int(x) for x in prompt_lengths.split(",")]
        result = self._npu.benchmark(str(name), prompt_lengths, int(max_tokens))
        if not result.success:
            raise DeviceFault(f"NPU benchmark failed: {result.error}")
        return result.value

    def info(self):
        return self._npu.info()
