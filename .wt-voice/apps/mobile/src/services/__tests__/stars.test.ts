import AsyncStorage from '@react-native-async-storage/async-storage';
import * as stars from '../stars';

const STORAGE_KEY = '@sloughgpt/starred-sessions';

beforeEach(async () => {
  jest.clearAllMocks();
  await AsyncStorage.clear();
  stars._resetCache();
});

describe('stars', () => {
  describe('getStarredIds', () => {
    it('returns empty array when no stars exist', async () => {
      const ids = await stars.getStarredIds();
      expect(ids).toEqual([]);
    });

    it('returns all starred session IDs', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(['session-1', 'session-2']));
      stars._resetCache();
      const ids = await stars.getStarredIds();
      expect(ids).toEqual(['session-1', 'session-2']);
    });
  });

  describe('starSession', () => {
    it('adds session to star list', async () => {
      await stars.starSession('session-1');
      const ids = await stars.getStarredIds();
      expect(ids).toEqual(['session-1']);
    });

    it('does not duplicate stars', async () => {
      await stars.starSession('session-1');
      await stars.starSession('session-1');
      const ids = await stars.getStarredIds();
      expect(ids).toEqual(['session-1']);
    });

    it('prepends new stars', async () => {
      await stars.starSession('session-1');
      await stars.starSession('session-2');
      const ids = await stars.getStarredIds();
      expect(ids).toEqual(['session-2', 'session-1']);
    });
  });

  describe('unstarSession', () => {
    it('removes session from star list', async () => {
      await stars.starSession('session-1');
      await stars.unstarSession('session-1');
      const ids = await stars.getStarredIds();
      expect(ids).toEqual([]);
    });

    it('does nothing for non-starred session', async () => {
      await stars.starSession('session-1');
      await stars.unstarSession('session-2');
      const ids = await stars.getStarredIds();
      expect(ids).toEqual(['session-1']);
    });
  });

  describe('isStarred', () => {
    it('returns true for starred session', async () => {
      await stars.starSession('session-1');
      expect(await stars.isStarred('session-1')).toBe(true);
    });

    it('returns false for non-starred session', async () => {
      expect(await stars.isStarred('session-1')).toBe(false);
    });
  });
});
