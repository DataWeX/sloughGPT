use crate::config::TrainConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn logsumexp_slice(slice: &[f32]) -> f32 {
    let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f32 = slice.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: usize,
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Vec<f32>,
}

impl OptimizerState {
    pub fn new(size: usize) -> Self {
        Self {
            step: 0,
            exp_avg: vec![0.0; size],
            exp_avg_sq: vec![0.0; size],
        }
    }
}

pub struct AdamW {
    config: TrainConfig,
    states: HashMap<String, OptimizerState>,
}

impl AdamW {
    pub fn new(config: &TrainConfig) -> Self {
        Self {
            config: config.clone(),
            states: HashMap::new(),
        }
    }

    pub fn register_param(&mut self, name: &str, size: usize) {
        self.states
            .insert(name.to_string(), OptimizerState::new(size));
    }

    pub fn step(&mut self, name: &str, params: &mut [f32], grads: &[f32], lr: f32) {
        if let Some(state) = self.states.get_mut(name) {
            state.step += 1;
            let step = state.step as f32;
            let (beta1, beta2, eps, weight_decay) = (
                self.config.beta1,
                self.config.beta2,
                self.config.eps,
                self.config.weight_decay,
            );

            for ((p, g), (m, v)) in params
                .iter_mut()
                .zip(grads.iter())
                .zip(state.exp_avg.iter_mut().zip(state.exp_avg_sq.iter_mut()))
            {
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * g * g;

                let m_hat = *m / (1.0 - beta1.powi(step as i32));
                let v_hat = *v / (1.0 - beta2.powi(step as i32));

                *p -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *p);
            }
        }
    }

    pub fn get_states(&self) -> &HashMap<String, OptimizerState> {
        &self.states
    }

    pub fn load_states(&mut self, states: HashMap<String, OptimizerState>) {
        self.states = states;
    }
}

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

pub fn compute_cross_entropy_loss(logits: &[f32], targets: &[i32], ignore_index: i32) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0usize;

    for (i, &target) in targets.iter().enumerate() {
        if target == ignore_index {
            continue;
        }
        let idx = target as usize;
        if idx < logits.len() {
            let log_prob = logits[idx] - logsumexp_slice(logits);
            loss -= log_prob;
            count += 1;
        }
    }

    loss / count as f32
}

pub fn fused_cross_entropy_backward(
    logits: &[f32],
    targets: &[i32],
    grad_output: f32,
    output: &mut [f32],
) {
    let n = targets.len();
    let vocab_size = logits.len() / n;

    for (i, &target) in targets.iter().enumerate() {
        let base = i * vocab_size;
        let max_logit = logits[base..base + vocab_size]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum_exp = 0.0f32;
        for j in 0..vocab_size {
            sum_exp += (logits[base + j] - max_logit).exp();
        }
        let log_sum_exp = max_logit + sum_exp.ln();

        for j in 0..vocab_size {
            let softmax = (logits[base + j] - max_logit).exp() / sum_exp;
            let grad = if j == target as usize {
                softmax - 1.0
            } else {
                softmax
            };
            output[base + j] += grad * grad_output / n as f32;
        }
    }
}
