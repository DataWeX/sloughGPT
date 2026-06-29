import {api, getApiUrl} from './api-client';
import {
  type ActivityPrediction,
  type ActivityStatus,
  type ActivityRecording,
  type SensorWindow,
} from '../types';

export async function recordData(body: SensorWindow) {
  return api.post<{id: number; path: string; samples: number}>(
    '/activity/data',
    body,
  );
}

export async function trainClassifier(opts?: {
  epochs?: number;
  lr?: number;
  batch_size?: number;
}) {
  return api.post<{
    status: string;
    epochs: number;
    final_loss: number | null;
    val_accuracy: number | null;
    num_samples: number;
    message: string;
  }>('/activity/train', opts || {});
}

export async function predict(body: {
  data: number[][];
}): Promise<ActivityPrediction> {
  return api.post<ActivityPrediction>('/activity/predict', body);
}

export async function getStatus(): Promise<ActivityStatus> {
  return api.get<ActivityStatus>('/activity/status');
}

export async function getDataset(): Promise<{
  recordings: ActivityRecording[];
  total: number;
}> {
  return api.get<{recordings: ActivityRecording[]; total: number}>(
    '/activity/dataset',
  );
}

export async function deleteAllData(): Promise<{deleted: number}> {
  return api.delete<{deleted: number}>('/activity/data');
}


