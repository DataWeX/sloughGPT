import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock the api-client module
jest.mock('../api-client', () => ({
  getApiUrl: jest.fn(async () => 'http://localhost:8000'),
  api: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    patch: jest.fn(),
    delete: jest.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));

import {api} from '../api-client';
import {
  recordData,
  trainClassifier,
  predict,
  getStatus,
  getDataset,
  deleteAllData,
} from '../activity-service';

beforeEach(() => {
  jest.clearAllMocks();
});

describe('activity-service', () => {
  describe('recordData', () => {
    it('posts sensor window to /activity/data', async () => {
      const mockResult = {id: 1, path: '/tmp/data.json', samples: 128};
      (api.post as jest.Mock).mockResolvedValue(mockResult);

      const window = {
        data: Array(128).fill([0.1, 0.2, 9.8, 0.01, 0.02, 0.03]),
        label: 1,
      };
      const result = await recordData(window);

      expect(api.post).toHaveBeenCalledWith('/activity/data', window);
      expect(result).toEqual(mockResult);
    });
  });

  describe('trainClassifier', () => {
    it('posts training options to /activity/train', async () => {
      const mockResult = {
        status: 'trained',
        epochs: 30,
        final_loss: 0.5,
        val_accuracy: 0.85,
        num_samples: 100,
        message: 'Training complete',
      };
      (api.post as jest.Mock).mockResolvedValue(mockResult);

      const result = await trainClassifier({epochs: 30, lr: 0.001, batch_size: 16});

      expect(api.post).toHaveBeenCalledWith('/activity/train', {epochs: 30, lr: 0.001, batch_size: 16});
      expect(result).toEqual(mockResult);
    });

    it('sends empty object when no opts', async () => {
      (api.post as jest.Mock).mockResolvedValue({
        status: 'trained', epochs: 10, final_loss: 1.0,
        val_accuracy: 0.5, num_samples: 50, message: 'ok',
      });

      await trainClassifier();

      expect(api.post).toHaveBeenCalledWith('/activity/train', {});
    });
  });

  describe('predict', () => {
    it('posts data to /activity/predict', async () => {
      const mockPrediction = {
        activity: 'walking',
        class_id: 1,
        confidence: 0.92,
        probabilities: [0.05, 0.92, 0.02, 0.005, 0.003, 0.002],
      };
      (api.post as jest.Mock).mockResolvedValue(mockPrediction);

      const data = Array(128).fill([0.1, 0.2, 9.8, 0.01, 0.02, 0.03]);
      const result = await predict({data});

      expect(api.post).toHaveBeenCalledWith('/activity/predict', {data});
      expect(result).toEqual(mockPrediction);
    });
  });

  describe('getStatus', () => {
    it('gets /activity/status', async () => {
      const mockStatus = {
        model_loaded: true,
        num_recordings: 42,
        num_labels: 30,
        activities: ['walking', 'running'],
        device: 'cpu',
      };
      (api.get as jest.Mock).mockResolvedValue(mockStatus);

      const result = await getStatus();

      expect(api.get).toHaveBeenCalledWith('/activity/status');
      expect(result).toEqual(mockStatus);
    });
  });

  describe('getDataset', () => {
    it('gets /activity/dataset', async () => {
      const mockDataset = {
        recordings: [
          {id: 1, path: '/tmp/1.json', samples: 128, label: 0, activity: 'stationary'},
          {id: 2, path: '/tmp/2.json', samples: 128, label: 1, activity: 'walking'},
        ],
        total: 2,
      };
      (api.get as jest.Mock).mockResolvedValue(mockDataset);

      const result = await getDataset();

      expect(api.get).toHaveBeenCalledWith('/activity/dataset');
      expect(result).toEqual(mockDataset);
    });
  });

  describe('deleteAllData', () => {
    it('deletes /activity/data', async () => {
      (api.delete as jest.Mock).mockResolvedValue({deleted: 42});

      const result = await deleteAllData();

      expect(api.delete).toHaveBeenCalledWith('/activity/data');
      expect(result).toEqual({deleted: 42});
    });
  });
});
