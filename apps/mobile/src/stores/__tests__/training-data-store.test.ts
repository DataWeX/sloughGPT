/**
 * Tests for training-data-store (Zustand store for on-device training pairs).
 */

import {useTrainingDataStore} from '../training-data-store';

beforeEach(() => {
  useTrainingDataStore.setState({pairs: {}, pairIds: [], hydrated: false, lastTrainResult: null});
});

describe('training-data-store', () => {
  describe('addPair', () => {
    it('adds a pair and returns an ID', () => {
      const id = useTrainingDataStore.getState().addPair('hello', 'hi', 's1');
      expect(id).toBeTruthy();
      expect(id).toMatch(/^tp_/);
    });

    it('stores correct fields', () => {
      const id = useTrainingDataStore.getState().addPair('user msg', 'assistant msg', 's1', 1);
      const pair = useTrainingDataStore.getState().pairs[id];
      expect(pair.user_msg).toBe('user msg');
      expect(pair.assistant_msg).toBe('assistant msg');
      expect(pair.session_id).toBe('s1');
      expect(pair.quality).toBe(1);
      expect(pair.synced).toBe(false);
      expect(pair.timestamp).toBeGreaterThan(0);
    });

    it('defaults quality to 0', () => {
      const id = useTrainingDataStore.getState().addPair('a', 'b', 's1');
      expect(useTrainingDataStore.getState().pairs[id].quality).toBe(0);
    });

    it('increments pairIds', () => {
      useTrainingDataStore.getState().addPair('a', 'b', 's1');
      useTrainingDataStore.getState().addPair('c', 'd', 's1');
      expect(useTrainingDataStore.getState().pairIds).toHaveLength(2);
    });
  });

  describe('updateQuality', () => {
    it('updates quality', () => {
      const id = useTrainingDataStore.getState().addPair('a', 'b', 's1');
      useTrainingDataStore.getState().updateQuality(id, 1);
      expect(useTrainingDataStore.getState().pairs[id].quality).toBe(1);
    });

    it('no-ops for nonexistent ID', () => {
      useTrainingDataStore.getState().addPair('a', 'b', 's1');
      useTrainingDataStore.getState().updateQuality('nonexistent', 1);
      expect(useTrainingDataStore.getState().pairIds).toHaveLength(1);
    });
  });

  describe('getPendingPairs', () => {
    it('returns unsynced pairs', () => {
      const store = useTrainingDataStore.getState();
      store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      expect(store.getPendingPairs()).toHaveLength(2);
    });

    it('excludes synced pairs', () => {
      const store = useTrainingDataStore.getState();
      const id = store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      store.markSynced([id]);
      expect(store.getPendingPairs()).toHaveLength(1);
    });

    it('sorts by timestamp ascending', () => {
      const store = useTrainingDataStore.getState();
      store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      const pending = store.getPendingPairs();
      expect(pending[0].timestamp).toBeLessThanOrEqual(pending[1].timestamp);
    });
  });

  describe('markSynced', () => {
    it('marks pairs as synced', () => {
      const store = useTrainingDataStore.getState();
      const id = store.addPair('a', 'b', 's1');
      store.markSynced([id]);
      expect(useTrainingDataStore.getState().pairs[id].synced).toBe(true);
    });

    it('updates pending count', () => {
      const store = useTrainingDataStore.getState();
      const id = store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      store.markSynced([id]);
      expect(useTrainingDataStore.getState().getPendingPairs()).toHaveLength(1);
    });
  });

  describe('clearSynced', () => {
    it('removes synced pairs from store', () => {
      const store = useTrainingDataStore.getState();
      const id = store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      store.markSynced([id]);
      useTrainingDataStore.getState().clearSynced();
      expect(useTrainingDataStore.getState().pairIds).toHaveLength(1);
    });
  });

  describe('clearAll', () => {
    it('removes all pairs', () => {
      const store = useTrainingDataStore.getState();
      store.addPair('a', 'b', 's1');
      store.addPair('c', 'd', 's1');
      store.clearAll();
      expect(store.pairIds).toHaveLength(0);
    });
  });

  describe('getStats', () => {
    it('returns correct stats', () => {
      const store = useTrainingDataStore.getState();
      store.addPair('a', 'b', 's1', 1);
      store.addPair('c', 'd', 's1', -1);
      store.addPair('e', 'f', 's1', 0);
      const stats = store.getStats();
      expect(stats.total).toBe(3);
      expect(stats.pending).toBe(3);
      expect(stats.synced).toBe(0);
      expect(stats.byQuality).toEqual({1: 1, '-1': 1, '0': 1});
    });
  });

  describe('setLastTrainResult', () => {
    it('stores training result', () => {
      useTrainingDataStore.getState().setLastTrainResult({
        loss: 1.2,
        checkpoint: 'ckpt_123',
        timestamp: Date.now(),
      });
      const result = useTrainingDataStore.getState().lastTrainResult;
      expect(result?.loss).toBe(1.2);
      expect(result?.checkpoint).toBe('ckpt_123');
    });
  });
});
