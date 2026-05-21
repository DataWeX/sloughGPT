use crate::config::TrainConfig;
use crate::optimizer::{AdamW, OptimizerState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[inline]
pub fn matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32], out: &mut [f32]) {
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    assert_eq!(out.len(), rows);

    for i in 0..rows {
        let mut sum = 0.0f32;
        let base = i * cols;

        let mut j = 0;
        let cols4 = (cols / 4) * 4;

        while j < cols4 {
            sum += mat[base + j] * vec[j];
            sum += mat[base + j + 1] * vec[j + 1];
            sum += mat[base + j + 2] * vec[j + 2];
            sum += mat[base + j + 3] * vec[j + 3];
            j += 4;
        }

        while j < cols {
            sum += mat[base + j] * vec[j];
            j += 1;
        }

        out[i] = sum;
    }
}

#[inline]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();

    for (i, &x) in input.iter().enumerate() {
        output[i] = (x / rms) * weight[i];
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn softmax(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut sum_exp = 0.0f32;
    for &x in input {
        sum_exp += (x - max_val).exp();
    }
    let log_sum_exp = max_val + sum_exp.ln();

    for i in 0..n {
        output[i] = ((input[i] - max_val).exp() / sum_exp).exp();
    }
}

#[inline]
pub fn cross_entropy_loss(logits: &[f32], targets: &[i32]) -> f32 {
    let batch_size = targets.len();
    let vocab_size = logits.len() / batch_size;
    let mut loss = 0.0f32;

    for (i, &target) in targets.iter().enumerate() {
        let base = i * vocab_size;

        let max_logit = logits[base..base + vocab_size]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let sum_exp: f32 = logits[base..base + vocab_size]
            .iter()
            .map(|&x| (x - max_logit).exp())
            .sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        loss -= logits[base + target as usize] - log_sum_exp;
    }

    loss / batch_size as f32
}

pub struct Gradients {
    pub grads: HashMap<String, Vec<f32>>,
}

impl Gradients {
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> &[f32] {
        self.grads.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        self.grads.get_mut(name)
    }

    pub fn insert(&mut self, name: String, grad: Vec<f32>) {
        self.grads.insert(name, grad);
    }

    pub fn add(&mut self, name: &str, grad: &[f32]) {
        if let Some(existing) = self.grads.get_mut(name) {
            for (i, &g) in grad.iter().enumerate() {
                existing[i] += g;
            }
        } else {
            self.grads.insert(name.to_string(), grad.to_vec());
        }
    }

    pub fn scale(&mut self, scale: f32) {
        for grad in self.grads.values_mut() {
            for g in grad.iter_mut() {
                *g *= scale;
            }
        }
    }

    pub fn clip(&mut self, max_norm: f32) {
        let mut total_norm = 0.0f32;
        for grad in self.grads.values() {
            for &g in grad {
                total_norm += g * g;
            }
        }
        total_norm = total_norm.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            self.scale(scale);
        }
    }
}

impl Default for Gradients {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
pub fn matvec_backward_input(
    weight: &[f32],
    rows: usize,
    cols: usize,
    grad_output: &[f32],
    grad_input: &mut [f32],
) {
    for j in 0..cols {
        let mut sum = 0.0f32;
        for i in 0..rows {
            let base = i * cols;
            sum += weight[base + j] * grad_output[i];
        }
        grad_input[j] = sum;
    }
}

#[inline]
pub fn matvec_backward_weight(
    input: &[f32],
    rows: usize,
    cols: usize,
    grad_output: &[f32],
) -> Vec<f32> {
    let mut grad_weight = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let base = i * cols;
        let go = grad_output[i];
        for j in 0..cols {
            grad_weight[base + j] = input[j] * go;
        }
    }
    grad_weight
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub token_embedding: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub attention: AttentionWeights,
    pub feed_forward: FeedForwardWeights,
}

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FeedForwardWeights {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

impl ModelWeights {
    pub fn new(config: &TrainConfig) -> Self {
        let dim = config.embedding_dim;
        let hidden_dim = config.hidden_dim;

        let token_embedding = vec![0.0; config.vocab_size * dim];
        let lm_head = vec![0.0; dim * config.vocab_size];

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerWeights {
                input_layernorm: vec![1.0; dim],
                post_attention_layernorm: vec![1.0; dim],
                attention: AttentionWeights {
                    q_proj: vec![0.0; dim * dim],
                    k_proj: vec![0.0; dim * dim],
                    v_proj: vec![0.0; dim * dim],
                    o_proj: vec![0.0; dim * dim],
                },
                feed_forward: FeedForwardWeights {
                    gate_proj: vec![0.0; hidden_dim * dim],
                    up_proj: vec![0.0; hidden_dim * dim],
                    down_proj: vec![0.0; dim * hidden_dim],
                },
            });
        }

        Self {
            token_embedding,
            lm_head,
            layers,
        }
    }

    pub fn from_hashmap(config: &TrainConfig, weights: &HashMap<String, Vec<f32>>) -> Self {
        let dim = config.embedding_dim;
        let hidden_dim = config.hidden_dim;

        let token_embedding = weights
            .get("token_embedding")
            .or_else(|| weights.get("token_embd.weight"))
            .cloned()
            .unwrap_or_else(|| vec![0.0; config.vocab_size * dim]);
        let lm_head = weights
            .get("lm_head")
            .or_else(|| weights.get("output.weight"))
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim * config.vocab_size]);

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layers_prefix = format!("layers.{}", layer_idx);
            let blk_prefix = format!("blk.{}", layer_idx);

            let input_layernorm = weights
                .get(&format!("{}.input_layernorm", layers_prefix))
                .or_else(|| weights.get(&format!("{}.attn_norm", blk_prefix)))
                .cloned()
                .unwrap_or_else(|| vec![1.0; dim]);

            let post_attention_layernorm = weights
                .get(&format!("{}.post_attention_layernorm", layers_prefix))
                .or_else(|| weights.get(&format!("{}.ffn_norm", blk_prefix)))
                .cloned()
                .unwrap_or_else(|| vec![1.0; dim]);

            layers.push(LayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention: AttentionWeights {
                    q_proj: weights
                        .get(&format!("{}.attn.q_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.attn_q", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; dim * dim]),
                    k_proj: weights
                        .get(&format!("{}.attn.k_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.attn_k", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; dim * dim]),
                    v_proj: weights
                        .get(&format!("{}.attn.v_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.attn_v", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; dim * dim]),
                    o_proj: weights
                        .get(&format!("{}.attn.o_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.attn_output", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; dim * dim]),
                },
                feed_forward: FeedForwardWeights {
                    gate_proj: weights
                        .get(&format!("{}.ffn.gate_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.ffn_gate", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; hidden_dim * dim]),
                    up_proj: weights
                        .get(&format!("{}.ffn.up_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.ffn_up", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; hidden_dim * dim]),
                    down_proj: weights
                        .get(&format!("{}.ffn.down_proj", layers_prefix))
                        .or_else(|| weights.get(&format!("{}.ffn_down", blk_prefix)))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; dim * hidden_dim]),
                },
            });
        }

        Self {
            token_embedding,
            lm_head,
            layers,
        }
    }

    pub fn to_hashmap(&self) -> HashMap<String, Vec<f32>> {
        let mut weights = HashMap::new();
        weights.insert("token_embedding".to_string(), self.token_embedding.clone());
        weights.insert("lm_head".to_string(), self.lm_head.clone());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", layer_idx);
            weights.insert(
                format!("{}.input_layernorm", prefix),
                layer.input_layernorm.clone(),
            );
            weights.insert(
                format!("{}.post_attention_layernorm", prefix),
                layer.post_attention_layernorm.clone(),
            );
            weights.insert(
                format!("{}.attn.q_proj", prefix),
                layer.attention.q_proj.clone(),
            );
            weights.insert(
                format!("{}.attn.k_proj", prefix),
                layer.attention.k_proj.clone(),
            );
            weights.insert(
                format!("{}.attn.v_proj", prefix),
                layer.attention.v_proj.clone(),
            );
            weights.insert(
                format!("{}.attn.o_proj", prefix),
                layer.attention.o_proj.clone(),
            );
            weights.insert(
                format!("{}.ffn.gate_proj", prefix),
                layer.feed_forward.gate_proj.clone(),
            );
            weights.insert(
                format!("{}.ffn.up_proj", prefix),
                layer.feed_forward.up_proj.clone(),
            );
            weights.insert(
                format!("{}.ffn.down_proj", prefix),
                layer.feed_forward.down_proj.clone(),
            );
        }

        weights
    }
}

pub struct Trainer {
    config: TrainConfig,
    model: ModelWeights,
    optimizer: AdamW,
    step: usize,
    gradients: Gradients,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        let model = ModelWeights::new(&config);
        let mut optimizer = AdamW::new(&config);

        optimizer.register_param("token_embedding", config.vocab_size * config.embedding_dim);
        optimizer.register_param("lm_head", config.embedding_dim * config.vocab_size);

        for layer_idx in 0..config.num_layers {
            let prefix = format!("layers.{}", layer_idx);
            optimizer.register_param(&format!("{}.input_layernorm", prefix), config.embedding_dim);
            optimizer.register_param(
                &format!("{}.post_attention_layernorm", prefix),
                config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.attn.q_proj", prefix),
                config.embedding_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.attn.k_proj", prefix),
                config.embedding_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.attn.v_proj", prefix),
                config.embedding_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.attn.o_proj", prefix),
                config.embedding_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.ffn.gate_proj", prefix),
                config.hidden_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.ffn.up_proj", prefix),
                config.hidden_dim * config.embedding_dim,
            );
            optimizer.register_param(
                &format!("{}.ffn.down_proj", prefix),
                config.embedding_dim * config.hidden_dim,
            );
        }

        Self {
            config,
            model,
            optimizer,
            step: 0,
            gradients: Gradients::new(),
        }
    }

    pub fn step(&mut self, batch: &[i32], targets: &[i32]) -> f32 {
        self.step += 1;
        let lr = self.get_lr();
        let loss = self.compute_gradients(batch, targets);
        self.optimizer_step(lr);
        loss
    }

    fn get_lr(&self) -> f32 {
        let step = self.step as f32;
        if step < self.config.warmup_steps as f32 {
            step / self.config.warmup_steps as f32 * self.config.learning_rate
        } else {
            let progress = (step - self.config.warmup_steps as f32)
                / (self.config.total_steps - self.config.warmup_steps) as f32;
            let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            cosine_decay * self.config.learning_rate
        }
    }

    pub fn forward(&self, tokens: &[i32]) -> Vec<f32> {
        let config = &self.config;
        let dim = config.embedding_dim;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        let mut hidden = Vec::with_capacity(tokens.len() * dim);
        for &token in tokens {
            let src = (token as usize) * dim;
            hidden.extend_from_slice(&self.model.token_embedding[src..src + dim]);
        }

        for layer in &self.model.layers {
            let mut normed = vec![0.0f32; dim];
            rms_norm(&hidden, &layer.input_layernorm, 1e-5, &mut normed);

            let mut q = vec![0.0f32; dim];
            let mut k = vec![0.0f32; dim];
            let mut v = vec![0.0f32; dim];
            matvec(&layer.attention.q_proj, dim, dim, &normed, &mut q);
            matvec(&layer.attention.k_proj, dim, dim, &normed, &mut k);
            matvec(&layer.attention.v_proj, dim, dim, &normed, &mut v);

            let mut attn_out = vec![0.0f32; dim];
            let scale = 1.0 / (dim as f32).sqrt();

            for i in 0..dim {
                let mut sum = 0.0f32;
                for j in 0..dim {
                    sum += q[i] * k[j] * scale;
                }
                attn_out[i] = sum / (dim as f32);
            }

            let mut proj_out = vec![0.0f32; dim];
            matvec(&layer.attention.o_proj, dim, dim, &attn_out, &mut proj_out);

            for i in 0..dim {
                hidden[i] += proj_out[i];
            }

            rms_norm(&hidden, &layer.post_attention_layernorm, 1e-5, &mut normed);

            let mut gate = vec![0.0f32; hidden_dim];
            let mut up = vec![0.0f32; hidden_dim];
            matvec(
                &layer.feed_forward.gate_proj,
                hidden_dim,
                dim,
                &normed,
                &mut gate,
            );
            matvec(
                &layer.feed_forward.up_proj,
                hidden_dim,
                dim,
                &normed,
                &mut up,
            );

            for i in 0..hidden_dim {
                gate[i] = silu(gate[i]) * up[i];
            }

            let mut down = vec![0.0f32; dim];
            matvec(
                &layer.feed_forward.down_proj,
                dim,
                hidden_dim,
                &gate,
                &mut down,
            );

            for i in 0..dim {
                hidden[i] += down[i];
            }
        }

        let mut final_hidden = vec![0.0f32; dim];
        rms_norm(
            &hidden,
            &self.model.token_embedding,
            1e-5,
            &mut final_hidden,
        );

        let mut logits = vec![0.0f32; vocab_size];
        matvec(
            &self.model.lm_head,
            vocab_size,
            dim,
            &final_hidden,
            &mut logits,
        );

        logits
    }

    fn compute_gradients(&mut self, batch: &[i32], targets: &[i32]) -> f32 {
        let config = &self.config;
        let dim = config.embedding_dim;
        let vocab_size = config.vocab_size;
        let batch_size = batch.len();

        self.gradients = Gradients::new();

        let logits = self.forward(batch);
        let loss = cross_entropy_loss(&logits, targets);

        let mut grad_logits = vec![0.0f32; logits.len()];
        for (i, &target) in targets.iter().enumerate() {
            let base = i * vocab_size;
            let max_logit = logits[base..base + vocab_size]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = logits[base..base + vocab_size]
                .iter()
                .map(|&x| (x - max_logit).exp())
                .sum();

            for j in 0..vocab_size {
                let softmax = (logits[base + j] - max_logit).exp() / sum_exp;
                grad_logits[base + j] = if j == target as usize {
                    (softmax - 1.0) / batch_size as f32
                } else {
                    softmax / batch_size as f32
                };
            }
        }

        let mut hidden_grad = vec![0.0f32; dim];
        matvec_backward_input(
            &self.model.lm_head,
            vocab_size,
            dim,
            &grad_logits,
            &mut hidden_grad,
        );
        let grad_lm_head = matvec_backward_weight(&hidden_grad, vocab_size, dim, &grad_logits);
        self.gradients.insert("lm_head".to_string(), grad_lm_head);

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let layer_prefix = format!("layers.{}", self.config.num_layers - 1 - layer_idx);

            let grad_o = matvec_backward_weight(&layer.attention.o_proj, dim, dim, &hidden_grad);
            self.gradients
                .add(&format!("{}.attn.o_proj", layer_prefix), &grad_o);

            let grad_q = matvec_backward_weight(&layer.attention.q_proj, dim, dim, &hidden_grad);
            self.gradients
                .add(&format!("{}.attn.q_proj", layer_prefix), &grad_q);

            let grad_k = matvec_backward_weight(&layer.attention.k_proj, dim, dim, &hidden_grad);
            self.gradients
                .add(&format!("{}.attn.k_proj", layer_prefix), &grad_k);

            let grad_v = matvec_backward_weight(&layer.attention.v_proj, dim, dim, &hidden_grad);
            self.gradients
                .add(&format!("{}.attn.v_proj", layer_prefix), &grad_v);

            hidden_grad = vec![0.0f32; dim];
        }

        self.gradients.clip(self.config.grad_clip);

        loss
    }

    fn optimizer_step(&mut self, lr: f32) {
        let weights = self.model.to_hashmap();
        for (name, weight) in weights {
            let grad = self.gradients.get(&name);
            let mut w = weight;
            self.optimizer.step(&name, &mut w, grad, lr);
        }
    }

    pub fn get_weights(&self) -> HashMap<String, Vec<f32>> {
        self.model.to_hashmap()
    }

    pub fn load_weights(&mut self, weights: HashMap<String, Vec<f32>>) {
        self.model = ModelWeights::from_hashmap(&self.config, &weights);
    }

    pub fn get_config(&self) -> TrainConfig {
        self.config.clone()
    }

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let state = CheckpointState {
            step: self.step,
            model: self.get_weights(),
            optim_states: self.optimizer.get_states().clone(),
        };
        let json = serde_json::to_string(&state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load_checkpoint(&mut self, path: &str) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let state: CheckpointState = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.step = state.step;
        self.load_weights(state.model);
        self.optimizer.load_states(state.optim_states);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct CheckpointState {
    step: usize,
    model: HashMap<String, Vec<f32>>,
    optim_states: HashMap<String, OptimizerState>,
}

#[inline]
fn rms_norm_backward_weight(_weight: &[f32], _input: &[f32], _grad_output: &mut [f32]) {
    // Simplified - just pass through gradient
    // Full implementation would compute dL/dweight
}
