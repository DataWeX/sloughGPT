import {api} from './api-client';
import {streamSSE, type SSEEvent} from './sse-client';

export interface TrainConfig {
  source_text?: string;
  dataset_id?: string;
  checkpoint_name?: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  soul_name: string;
  algo: string;
}

export interface TrainingStatus {
  running: boolean;
  config: Record<string, unknown>;
}

export interface Checkpoint {
  name: string;
  soul: string;
  loss: number | null;
  steps: number;
  traits: Record<string, number>;
  created_at: string;
  size_mb?: number;
  verdict?: string;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  file_count: number;
  total_chars: number;
}

export interface FineTunedModel {
  name: string;
  model_path: string;
  size_mb: number;
  size_bytes?: number;
  created_at?: string;
  model: string;
  dataset: string;
  model_name?: string;
  final_loss?: number | null;
  epochs?: number;
}

export interface TrainingJob {
  job_id?: string;
  id?: string;
  name?: string;
  status: string;
  phase?: string;
  progress?: number;
  model?: string;
  dataset?: string;
  method?: string;
  epochs?: number;
  current_epoch?: number;
  global_step?: number;
  total_steps?: number;
  steps_per_sec?: number;
  eta_s?: number | null;
  elapsed_s?: number;
  loss?: number;
  train_loss?: number;
  eval_loss?: number;
  checkpoint?: string;
  data_source?: string;
  result?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  output_dir?: string;
  sou_path?: string;
  epochs_completed?: number;
  error?: string;
  status_message?: string;
}

// ── Auto-train ─────────────────────────────────────────────────────────────

export async function startTraining(config: TrainConfig) {
  return api.post<{status: string; data_path: string; epochs: number}>(
    '/training/start',
    config,
  );
}

export async function stopTraining() {
  return api.post<{status: string}>('/training/stop');
}

export async function getTrainingStatus() {
  return api.get<TrainingStatus>('/training/status');
}

export async function listCheckpoints() {
  return api.get<Checkpoint[]>('/training/checkpoints');
}

export async function deleteCheckpoint(name: string) {
  return api.delete(`/training/checkpoints/${name}`);
}

export async function loadCheckpoint(name: string) {
  return api.post(`/training/checkpoints/${name}/load`);
}

export async function* streamTraining(
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  yield* streamSSE('/training/stream', {}, signal);
}

// ── Datasets ───────────────────────────────────────────────────────────────

export async function listDatasets() {
  return api.get<Dataset[]>('/datasets');
}

export async function importDatasetUrl(url: string, name?: string) {
  return api.post<{files_imported: number; total_chars: number}>(
    '/datasets/import/url',
    {url, name},
  );
}

export async function importDatasetGithub(repo: string, name?: string) {
  return api.post<{files_imported: number; total_chars: number}>(
    '/datasets/import/github',
    {repo, name},
  );
}

export async function importDatasetHuggingface(dataset: string, name?: string) {
  return api.post<{files_imported: number; total_chars: number}>(
    '/datasets/import/huggingface',
    {dataset, name},
  );
}

export async function importDatasetCsv(url: string, name?: string) {
  return api.post<{files_imported: number; total_chars: number}>(
    '/datasets/import/csv',
    {url, name},
  );
}

// ── Training Jobs ──────────────────────────────────────────────────────────

export async function listTrainingJobs() {
  return api.get<TrainingJob[]>('/training/jobs');
}

export async function getTrainingJob(jobId: string) {
  return api.get<TrainingJob>(`/training/jobs/${jobId}`);
}

export async function stopTrainingJob(jobId: string) {
  return api.post(`/training/jobs/${jobId}/stop`);
}

export async function deleteTrainingJob(jobId: string) {
  return api.delete(`/training/jobs/${jobId}`);
}

export async function getJobSummary(jobId: string) {
  return api.get<{summary: string}>(`/training/jobs/${jobId}/summary`);
}

// ── Fine-tuned Models ──────────────────────────────────────────────────────

export async function listFineTunedModels() {
  return api.get<{models: FineTunedModel[]}>('/training/finetuned-models');
}

export async function loadFineTunedModel(name: string) {
  return api.post(`/training/finetuned-models/${name}/load`);
}

export async function deleteFineTunedModel(name: string) {
  return api.delete(`/training/finetuned-models/${name}`);
}

// ── LoRA Fine-tune ─────────────────────────────────────────────────────────

export async function startLoraFinetune(opts: {
  model_path: string;
  dataset: string;
  rank?: number;
  alpha?: number;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
}) {
  return api.post<{job_id: string}>('/training/lora-finetune', opts);
}

// ── Adapters ───────────────────────────────────────────────────────────────

export async function loadAdapter(path: string, merge: boolean = false) {
  return api.post('/training/load-adapter', {path, merge});
}

export async function unloadAdapter() {
  return api.post('/training/unload-adapter');
}
