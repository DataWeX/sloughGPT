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

export async function startTraining(config: TrainConfig) {
  return api.post<{status: string; data_path: string; epochs: number}>(
    '/auto-train/start',
    config,
  );
}

export async function stopTraining() {
  return api.post<{status: string}>('/auto-train/stop');
}

export async function getTrainingStatus() {
  return api.get<TrainingStatus>('/auto-train/status');
}

export async function listCheckpoints() {
  return api.get<Checkpoint[]>('/auto-train/checkpoints');
}

export async function deleteCheckpoint(name: string) {
  return api.delete(`/auto-train/checkpoints/${name}`);
}

export async function loadCheckpoint(name: string) {
  return api.post(`/auto-train/checkpoints/${name}/load`);
}

export async function listDatasets() {
  return api.get<Dataset[]>('/datasets');
}

export async function* streamTraining(
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  yield* streamSSE('/auto-train/stream', {}, signal);
}

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

export async function listTrainingJobs() {
  return api.get<any[]>('/training/jobs');
}
