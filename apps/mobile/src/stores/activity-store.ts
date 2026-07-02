import {create} from 'zustand';
import {
  recordData,
  trainClassifier,
  predict,
  getStatus,
  getDataset,
  deleteAllData,
} from '../services/activity-service';
import {
  predictLocal,
  initInference,
  isLocalModelReady,
} from '../services/activity-inference';
import {
  type SensorReading,
  type ActivityPrediction,
  type ActivityRecording,
} from '../types';
import type {ModelSyncStatus} from '../services/model-sync';

export type ActivityPhase =
  | 'idle'
  | 'recording'
  | 'training'
  | 'complete'
  | 'failed';

export interface LossPoint {
  epoch: number;
  loss: number;
  valLoss: number;
  valAccuracy: number;
}

interface ActivityState {
  // Sensor data
  sensorHistory: SensorReading[];
  isRecording: boolean;
  recordingLabel: number | null;

  // Training
  phase: ActivityPhase;
  trainingEpochs: number;
  trainingEpoch: number;
  trainingLoss: number | null;
  trainingAccuracy: number | null;
  lossHistory: LossPoint[];
  numSamples: number;
  error: string | null;

  // Status
  modelLoaded: boolean;
  dataset: ActivityRecording[];
  totalRecordings: number;
  modelSync: ModelSyncStatus | null;

  // Background recording
  backgroundRecording: boolean;
  bgBufferSize: number;
  bgLastSync: number | null;

  // Prediction
  lastPrediction: ActivityPrediction | null;

  // Actions
  pushReading: (r: SensorReading) => void;
  clearHistory: () => void;
  startRecording: (label?: number) => void;
  stopRecording: () => SensorReading[];
  setRecordingLabel: (label: number | null) => void;
  startTraining: (opts?: {epochs?: number}) => Promise<void>;
  startTrainingStream: (opts?: {epochs?: number}) => Promise<void>;
  setTrainingProgress: (p: {epoch: number; epochs: number; loss: number; val_loss: number; val_accuracy: number}) => void;
  predictActivity: (data: number[][]) => Promise<ActivityPrediction | null>;
  refreshStatus: () => Promise<void>;
  setModelSync: (s: ModelSyncStatus) => void;
  setBackgroundState: (active: boolean, bufferSize?: number, lastSync?: number | null) => void;
  clearError: () => void;
  deleteAll: () => Promise<void>;
  reset: () => void;
}

const WINDOW_SIZE = 128;

const initialState = {
  sensorHistory: [] as SensorReading[],
  isRecording: false,
  recordingLabel: null as number | null,
  phase: 'idle' as ActivityPhase,
  trainingEpochs: 0,
  trainingEpoch: 0,
  trainingLoss: null as number | null,
  trainingAccuracy: null as number | null,
  lossHistory: [] as LossPoint[],
  numSamples: 0,
  error: null as string | null,
  modelLoaded: false,
  dataset: [] as ActivityRecording[],
  totalRecordings: 0,
  modelSync: null as ModelSyncStatus | null,
  backgroundRecording: false,
  bgBufferSize: 0,
  bgLastSync: null as number | null,
  lastPrediction: null as ActivityPrediction | null,
};

export const useActivityStore = create<ActivityState>((set, get) => ({
  ...initialState,

  pushReading: r =>
    set(s => {
      const history = [...s.sensorHistory, r];
      if (history.length > WINDOW_SIZE * 2) {
        return {sensorHistory: history.slice(-WINDOW_SIZE)};
      }
      return {sensorHistory: history};
    }),

  clearHistory: () => set({sensorHistory: []}),

  startRecording: (label?: number) =>
    set({
      isRecording: true,
      recordingLabel: label ?? null,
      sensorHistory: [],
      lastPrediction: null,
    }),

  stopRecording: () => {
    const window = get().sensorHistory.slice(-WINDOW_SIZE);
    set({isRecording: false});
    return window;
  },

  setRecordingLabel: label => set({recordingLabel: label}),

  startTraining: async (opts?: {epochs?: number}) => {
    set({phase: 'training', error: null, trainingLoss: null, trainingAccuracy: null, lossHistory: []});
    try {
      const result = await trainClassifier({
        epochs: opts?.epochs ?? 30,
        lr: 0.001,
        batch_size: 16,
      });
      set({
        phase: 'complete',
        trainingEpochs: result.epochs,
        trainingLoss: result.final_loss,
        trainingAccuracy: result.val_accuracy,
        numSamples: result.num_samples,
      });
      get().refreshStatus();
    } catch (err: any) {
      set({phase: 'failed', error: err.message || 'Training failed'});
    }
  },

  startTrainingStream: async (opts?: {epochs?: number}) => {
    set({phase: 'training', error: null, trainingLoss: null, trainingAccuracy: null, lossHistory: []});
    try {
      const {getApiUrl} = require('../services/api-client');
      const baseUrl = await getApiUrl();
      const epochs = opts?.epochs ?? 30;
      const res = await fetch(`${baseUrl}/activity/train/stream`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({epochs, lr: 0.001, batch_size: 16}),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Training error: ${text}`);
      }
      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(trimmed.slice(6));
            if (data.status === 'complete') {
              set({
                phase: 'complete',
                trainingEpochs: data.epochs || epochs,
                trainingAccuracy: data.val_accuracy,
                numSamples: data.num_samples || 0,
              });
              get().refreshStatus();
              return;
            }
            if (data.status === 'error') {
              set({phase: 'failed', error: data.message || 'Training failed'});
              return;
            }
            // Epoch progress
            set(s => ({
              trainingEpoch: data.epoch || 0,
              trainingEpochs: data.epochs || epochs,
              trainingLoss: data.loss ?? s.trainingLoss,
              trainingAccuracy: data.val_accuracy ?? s.trainingAccuracy,
              lossHistory: [
                ...s.lossHistory,
                {
                  epoch: data.epoch || 0,
                  loss: data.loss || 0,
                  valLoss: data.val_loss || 0,
                  valAccuracy: data.val_accuracy || 0,
                },
              ],
            }));
          } catch {}
        }
      }
      set({phase: 'complete'});
      get().refreshStatus();
    } catch (err: any) {
      set({phase: 'failed', error: err.message || 'Training failed'});
    }
  },

  setTrainingProgress: p =>
    set(s => ({
      trainingEpoch: p.epoch,
      trainingEpochs: p.epochs,
      trainingLoss: p.loss,
      trainingAccuracy: p.val_accuracy,
      lossHistory: [
        ...s.lossHistory,
        {
          epoch: p.epoch,
          loss: p.loss,
          valLoss: p.val_loss,
          valAccuracy: p.val_accuracy,
        },
      ],
    })),

  predictActivity: async (data: number[][]) => {
    // Try local inference first (no server round-trip)
    const localResult = await predictLocal(data);
    if (localResult) {
      const prediction: ActivityPrediction = {
        class_id: localResult.classId,
        activity: localResult.className,
        confidence: Math.max(...localResult.probabilities),
        probabilities: localResult.probabilities,
      };
      set({lastPrediction: prediction});
      return prediction;
    }

    // Fallback to server inference
    try {
      const result = await predict({data});
      set({lastPrediction: result});
      return result;
    } catch (err: any) {
      set({error: err.message || 'Prediction failed'});
      return null;
    }
  },

  refreshStatus: async () => {
    try {
      const [status, ds] = await Promise.all([
        getStatus().catch(() => null),
        getDataset().catch(() => null),
      ]);
      if (status) {
        set({modelLoaded: status.model_loaded});
      }
      if (ds) {
        set({dataset: ds.recordings, totalRecordings: ds.total});
      }
    } catch {}
  },

  setModelSync: s => set({modelSync: s}),
  setBackgroundState: (active, bufferSize, lastSync) =>
    set({
      backgroundRecording: active,
      bgBufferSize: bufferSize ?? 0,
      bgLastSync: lastSync ?? null,
    }),

  clearError: () => set({error: null}),

  deleteAll: async () => {
    try {
      await deleteAllData();
      set({dataset: [], totalRecordings: 0, modelLoaded: false, lastPrediction: null});
    } catch (err: any) {
      set({error: err.message || 'Delete failed'});
    }
  },

  reset: () => set(initialState),
}));
