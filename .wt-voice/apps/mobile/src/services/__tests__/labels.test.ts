import AsyncStorage from '@react-native-async-storage/async-storage';
import * as labels from '../labels';

const STORAGE_KEY = '@sloughgpt/session-labels';

beforeEach(async () => {
  jest.clearAllMocks();
  await AsyncStorage.clear();
  labels._resetCache();
});

describe('labels', () => {
  describe('getLabels', () => {
    it('returns empty array when no labels exist', async () => {
      const result = await labels.getLabels('session-1');
      expect(result).toEqual([]);
    });

    it('returns labels for a session', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ 'session-1': ['work', 'urgent'] }));
      labels._resetCache();
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work', 'urgent']);
    });

    it('returns empty for session with no labels', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ 'session-1': ['work'] }));
      labels._resetCache();
      const result = await labels.getLabels('session-2');
      expect(result).toEqual([]);
    });
  });

  describe('setLabels', () => {
    it('sets labels for a session', async () => {
      await labels.setLabels('session-1', ['work', 'urgent']);
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work', 'urgent']);
    });

    it('deduplicates labels', async () => {
      await labels.setLabels('session-1', ['work', 'work', 'urgent']);
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work', 'urgent']);
    });

    it('trims whitespace from labels', async () => {
      await labels.setLabels('session-1', ['  work  ', 'urgent']);
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work', 'urgent']);
    });

    it('removes session entry when empty array', async () => {
      await labels.setLabels('session-1', ['work']);
      await labels.setLabels('session-1', []);
      labels._resetCache();
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      expect(JSON.parse(raw!)).toEqual({});
    });

    it('overwrites existing labels', async () => {
      await labels.setLabels('session-1', ['work']);
      await labels.setLabels('session-1', ['personal']);
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['personal']);
    });
  });

  describe('addLabel', () => {
    it('adds a label to a session', async () => {
      await labels.addLabel('session-1', 'work');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work']);
    });

    it('does not duplicate labels', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.addLabel('session-1', 'work');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work']);
    });

    it('ignores empty label', async () => {
      await labels.addLabel('session-1', '');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual([]);
    });

    it('does not create entry for empty label', async () => {
      await labels.addLabel('session-1', '   ');
      labels._resetCache();
      expect(await AsyncStorage.getItem(STORAGE_KEY)).toBeNull();
    });

    it('keeps existing labels when adding new one', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.addLabel('session-1', 'urgent');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work', 'urgent']);
    });
  });

  describe('removeLabel', () => {
    it('removes a label from a session', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.removeLabel('session-1', 'work');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual([]);
    });

    it('does nothing for non-existent label', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.removeLabel('session-1', 'urgent');
      const result = await labels.getLabels('session-1');
      expect(result).toEqual(['work']);
    });

    it('removes session entry when last label removed', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.removeLabel('session-1', 'work');
      labels._resetCache();
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      expect(JSON.parse(raw!)).toEqual({});
    });
  });

  describe('getAllDistinctLabels', () => {
    it('returns empty array when no labels exist', async () => {
      const all = await labels.getAllDistinctLabels();
      expect(all).toEqual([]);
    });

    it('returns all distinct labels across sessions', async () => {
      await labels.addLabel('session-1', 'work');
      await labels.addLabel('session-1', 'urgent');
      await labels.addLabel('session-2', 'work');
      await labels.addLabel('session-2', 'personal');
      const all = await labels.getAllDistinctLabels();
      expect(all).toEqual(['personal', 'urgent', 'work']);
    });
  });
});
