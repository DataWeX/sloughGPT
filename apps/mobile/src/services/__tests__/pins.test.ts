import AsyncStorage from '@react-native-async-storage/async-storage';
import * as pins from '../pins';

const STORAGE_KEY = '@sloughgpt/pinned-messages';

beforeEach(async () => {
  jest.clearAllMocks();
  await AsyncStorage.clear();
  pins._resetCache();
});

describe('pins', () => {
  describe('getPinnedIds', () => {
    it('returns empty array when no pins exist', async () => {
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual([]);
    });

    it('returns pins for a session', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ 'session-1': ['msg-1', 'msg-2'] }));
      pins._resetCache();
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual(['msg-1', 'msg-2']);
    });

    it('returns empty array for session with no pins', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ 'session-1': ['msg-1'] }));
      pins._resetCache();
      const ids = await pins.getPinnedIds('session-2');
      expect(ids).toEqual([]);
    });
  });

  describe('pinMessage', () => {
    it('pins a message in a session', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual(['msg-1']);
    });

    it('does not duplicate pins', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      await pins.pinMessage('session-1', 'msg-1');
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual(['msg-1']);
    });

    it('prepends new pins', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      await pins.pinMessage('session-1', 'msg-2');
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual(['msg-2', 'msg-1']);
    });
  });

  describe('unpinMessage', () => {
    it('removes a pinned message', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      await pins.unpinMessage('session-1', 'msg-1');
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual([]);
    });

    it('does nothing for unpinned message', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      await pins.unpinMessage('session-1', 'msg-2');
      const ids = await pins.getPinnedIds('session-1');
      expect(ids).toEqual(['msg-1']);
    });

    it('cleans up session entry when last pin removed', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      pins._resetCache();
      const rawBefore = await AsyncStorage.getItem(STORAGE_KEY);
      expect(JSON.parse(rawBefore!)).toEqual({ 'session-1': ['msg-1'] });

      await pins.unpinMessage('session-1', 'msg-1');
      pins._resetCache();
      const rawAfter = await AsyncStorage.getItem(STORAGE_KEY);
      expect(JSON.parse(rawAfter!)).toEqual({});
    });
  });

  describe('isPinned', () => {
    it('returns true for pinned message', async () => {
      await pins.pinMessage('session-1', 'msg-1');
      expect(await pins.isPinned('session-1', 'msg-1')).toBe(true);
    });

    it('returns false for unpinned message', async () => {
      expect(await pins.isPinned('session-1', 'msg-1')).toBe(false);
    });
  });
});
