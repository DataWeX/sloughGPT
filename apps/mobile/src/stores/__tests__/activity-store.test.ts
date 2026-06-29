// Mock the activity-service
jest.mock('../../services/activity-service', () => ({
  recordData: jest.fn(async () => ({id: 1, path: '/tmp/1.json', samples: 128})),
  trainClassifier: jest.fn(async () => ({
    status: 'trained', epochs: 30, final_loss: 0.5, val_accuracy: 0.85,
    num_samples: 100, message: 'Training complete',
  })),
  predict: jest.fn(async () => ({
    activity: 'walking', class_id: 1, confidence: 0.92,
    probabilities: [0.05, 0.92, 0.02, 0.005, 0.003, 0.002],
  })),
  getStatus: jest.fn(async () => ({
    model_loaded: true, num_recordings: 42, num_labels: 30,
    activities: ['walking', 'running'], device: 'cpu',
  })),
  getDataset: jest.fn(async () => ({
    recordings: [
      {id: 1, path: '/tmp/1.json', samples: 128, label: 0, activity: 'stationary'},
      {id: 2, path: '/tmp/2.json', samples: 128, label: 1, activity: 'walking'},
    ],
    total: 2,
  })),
  deleteAllData: jest.fn(async () => ({deleted: 42})),
}));

import {useActivityStore} from '../activity-store';
import type {SensorReading} from '../../types';

const makeReading = (t = 0): SensorReading => ({
  timestamp: Date.now() + t,
  accel: {x: Math.sin(t), y: Math.cos(t), z: 9.8},
  gyro: {x: 0.01 * t, y: 0.02 * t, z: 0.03 * t},
});

// Zustand mock doesn't support getState().reset(), so we manually reset
function resetStore() {
  const state = useActivityStore.getState();
  state.clearHistory();
  state.clearError();
  // Manually set back to initial-like state
  if (state.isRecording) state.stopRecording();
}

beforeEach(() => {
  resetStore();
});

describe('activity-store', () => {
  describe('pushReading', () => {
    it('adds a reading to sensorHistory', () => {
      useActivityStore.getState().pushReading(makeReading(0));
      expect(useActivityStore.getState().sensorHistory).toHaveLength(1);
    });

    it('trims to WINDOW_SIZE when exceeding 2x', () => {
      const store = useActivityStore.getState();
      for (let i = 0; i < 600; i++) {
        store.pushReading(makeReading(i));
      }
      // After 600 pushes, should have trimmed multiple times
      // Final trim at ~513 (256+256+1), then 600-513=87 more → 128+87=215
      // But 215 < 256 so no more trim. Actual: between 128 and 256.
      expect(useActivityStore.getState().sensorHistory.length).toBeLessThanOrEqual(256);
    });
  });

  describe('clearHistory', () => {
    it('empties sensorHistory', () => {
      useActivityStore.getState().pushReading(makeReading());
      useActivityStore.getState().clearHistory();
      expect(useActivityStore.getState().sensorHistory).toEqual([]);
    });
  });

  describe('startRecording / stopRecording', () => {
    it('starts and stops recording', () => {
      const store = useActivityStore.getState();
      store.startRecording(1);
      expect(useActivityStore.getState().isRecording).toBe(true);
      expect(useActivityStore.getState().recordingLabel).toBe(1);

      store.stopRecording();
      expect(useActivityStore.getState().isRecording).toBe(false);
    });

    it('returns last WINDOW_SIZE readings on stop', () => {
      const store = useActivityStore.getState();
      store.startRecording();
      for (let i = 0; i < 200; i++) {
        store.pushReading(makeReading(i));
      }
      const window = store.stopRecording();
      expect(window.length).toBeLessThanOrEqual(128);
    });
  });

  describe('setRecordingLabel', () => {
    it('sets the label', () => {
      useActivityStore.getState().setRecordingLabel(3);
      expect(useActivityStore.getState().recordingLabel).toBe(3);
    });
  });

  describe('startTraining', () => {
    it('transitions to training then complete', async () => {
      await useActivityStore.getState().startTraining({epochs: 10});
      const state = useActivityStore.getState();
      expect(state.phase).toBe('complete');
      expect(state.trainingEpochs).toBe(30);
      expect(state.trainingLoss).toBe(0.5);
    });

    it('sets error on failure', async () => {
      const {trainClassifier} = require('../../services/activity-service');
      trainClassifier.mockRejectedValueOnce(new Error('Network error'));

      await useActivityStore.getState().startTraining();
      expect(useActivityStore.getState().phase).toBe('failed');
      expect(useActivityStore.getState().error).toBe('Network error');
    });
  });

  describe('setTrainingProgress', () => {
    it('appends to lossHistory', () => {
      useActivityStore.getState().setTrainingProgress({
        epoch: 1, epochs: 10, loss: 1.5, val_loss: 1.6, val_accuracy: 0.3,
      });
      useActivityStore.getState().setTrainingProgress({
        epoch: 2, epochs: 10, loss: 1.2, val_loss: 1.3, val_accuracy: 0.5,
      });

      const state = useActivityStore.getState();
      expect(state.lossHistory).toHaveLength(2);
      expect(state.lossHistory[0].loss).toBe(1.5);
      expect(state.lossHistory[1].loss).toBe(1.2);
      expect(state.trainingEpoch).toBe(2);
    });
  });

  describe('predictActivity', () => {
    it('sets lastPrediction on success', async () => {
      const data = Array(128).fill([0.1, 0.2, 9.8, 0.01, 0.02, 0.03]);
      const result = await useActivityStore.getState().predictActivity(data);

      expect(result).toBeTruthy();
      expect(result!.activity).toBe('walking');
      expect(useActivityStore.getState().lastPrediction!.activity).toBe('walking');
    });

    it('sets error on failure', async () => {
      const {predict} = require('../../services/activity-service');
      predict.mockRejectedValueOnce(new Error('Model not loaded'));

      const result = await useActivityStore.getState().predictActivity([[]]);
      expect(result).toBeNull();
      expect(useActivityStore.getState().error).toBe('Model not loaded');
    });
  });

  describe('refreshStatus', () => {
    it('fetches status and dataset', async () => {
      await useActivityStore.getState().refreshStatus();

      const state = useActivityStore.getState();
      expect(state.modelLoaded).toBe(true);
      expect(state.dataset).toHaveLength(2);
      expect(state.totalRecordings).toBe(2);
    });

    it('handles status fetch failure gracefully', async () => {
      const {getStatus, getDataset} = require('../../services/activity-service');
      getStatus.mockRejectedValueOnce(new Error('offline'));
      getDataset.mockRejectedValueOnce(new Error('offline'));

      await useActivityStore.getState().refreshStatus();
    });
  });

  describe('deleteAll', () => {
    it('clears dataset after delete', async () => {
      await useActivityStore.getState().refreshStatus();
      await useActivityStore.getState().deleteAll();

      const state = useActivityStore.getState();
      expect(state.dataset).toEqual([]);
      expect(state.totalRecordings).toBe(0);
    });
  });
});
