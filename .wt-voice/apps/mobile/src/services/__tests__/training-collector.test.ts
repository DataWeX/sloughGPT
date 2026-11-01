/**
 * Tests for training data collector (Zustand store-backed).
 */

jest.mock('@react-native-async-storage/async-storage', () => {
  const store = new Map<string, string>();
  return {
    __esModule: true,
    default: {
      getItem: jest.fn((key: string) => Promise.resolve(store.get(key) || null)),
      setItem: jest.fn((key: string, val: string) => { store.set(key, val); return Promise.resolve(); }),
      removeItem: jest.fn((key: string) => { store.delete(key); return Promise.resolve(); }),
      clear: jest.fn(() => { store.clear(); return Promise.resolve(); }),
    },
  };
});

const mockMobileTrain = jest.fn().mockResolvedValue({
  success: true, checkpoint_name: 'test_ckpt', loss: 1.2, steps: 10, elapsed_ms: 500,
});
jest.mock('../api-client', () => ({
  api: {mobileTrain: (...args: any[]) => mockMobileTrain(...args), get: jest.fn()},
}));

import {useTrainingDataStore, hydrateTrainingData} from '../../stores/training-data-store';
import {
  initCollector,
  collectPair,
  getPendingPairs,
  getStats,
  clearPending,
  triggerTraining,
  updateQuality,
} from '../training-collector';

beforeEach(() => {
  jest.clearAllMocks();
  useTrainingDataStore.setState({pairs: {}, pairIds: [], hydrated: false, lastTrainResult: null});
});

describe('training-collector', () => {
  describe('collectPair', () => {
    it('adds a pair to the store', () => {
      collectPair('hello', 'hi there', 's1');
      expect(getPendingPairs()).toHaveLength(1);
    });

    it('stores correct fields', () => {
      collectPair('user msg', 'assistant msg', 's1', 1);
      const pairs = getPendingPairs();
      expect(pairs[0].user_msg).toBe('user msg');
      expect(pairs[0].assistant_msg).toBe('assistant msg');
      expect(pairs[0].session_id).toBe('s1');
      expect(pairs[0].quality).toBe(1);
      expect(pairs[0].synced).toBe(false);
    });

    it('increments pending count', () => {
      collectPair('a', 'b', 's1');
      collectPair('c', 'd', 's1');
      expect(getStats().pending).toBe(2);
    });

    it('returns a pair ID', () => {
      const id = collectPair('a', 'b', 's1');
      expect(id).toBeTruthy();
      expect(id).toMatch(/^tp_/);
    });
  });

  describe('updateQuality', () => {
    it('updates quality on a pair', () => {
      const id = collectPair('a', 'b', 's1');
      updateQuality(id, 1);
      expect(useTrainingDataStore.getState().pairs[id].quality).toBe(1);
    });

    it('no-ops for nonexistent id', () => {
      collectPair('a', 'b', 's1');
      updateQuality('nonexistent', 1);
      expect(getStats().pending).toBe(1);
    });
  });

  describe('clearPending', () => {
    it('removes all synced pairs', () => {
      collectPair('a', 'b', 's1');
      collectPair('c', 'd', 's1');
      const store = useTrainingDataStore.getState();
      store.markSynced(store.pairIds);
      clearPending();
      expect(getPendingPairs()).toHaveLength(0);
    });
  });

  describe('triggerTraining', () => {
    it('returns null if fewer than minPairs', async () => {
      collectPair('a', 'b', 's1');
      const result = await triggerTraining('base_ckpt', 10);
      expect(result).toBeNull();
      expect(mockMobileTrain).not.toHaveBeenCalled();
    });

    it('calls mobileTrain when enough pairs', async () => {
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      const result = await triggerTraining('base_ckpt', 10);
      expect(mockMobileTrain).toHaveBeenCalledTimes(1);
      expect(result?.success).toBe(true);
    });

    it('marks pairs as synced after success', async () => {
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      await triggerTraining('base_ckpt', 10);
      expect(getPendingPairs()).toHaveLength(0);
    });

    it('stores train result', async () => {
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      await triggerTraining('base_ckpt', 10);
      const result = useTrainingDataStore.getState().lastTrainResult;
      expect(result?.loss).toBe(1.2);
      expect(result?.checkpoint).toBe('test_ckpt');
    });

    it('propagates server errors', async () => {
      mockMobileTrain.mockRejectedValueOnce(new Error('server down'));
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      await expect(triggerTraining('base_ckpt', 10)).rejects.toThrow('server down');
    });

    it('does not mark synced on failure', async () => {
      mockMobileTrain.mockRejectedValueOnce(new Error('fail'));
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      await expect(triggerTraining('base_ckpt', 10)).rejects.toThrow();
      expect(getPendingPairs()).toHaveLength(10);
    });
  });

  describe('stats', () => {
    it('tracks total, pending, synced', () => {
      collectPair('a', 'b', 's1');
      collectPair('c', 'd', 's1');
      const stats = getStats();
      expect(stats.total).toBe(2);
      expect(stats.pending).toBe(2);
      expect(stats.synced).toBe(0);
    });

    it('updates after sync', async () => {
      for (let i = 0; i < 10; i++) collectPair(`u${i}`, `a${i}`, 's1');
      await triggerTraining('base_ckpt', 10);
      const stats = getStats();
      expect(stats.total).toBe(10);
      expect(stats.pending).toBe(0);
      expect(stats.synced).toBe(10);
    });
  });
});
