import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/bookmarks';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {
  getBookmarks,
  addBookmark,
  removeBookmark,
  isBookmarked,
  _resetCache,
} = require('../bookmarks');

beforeEach(() => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  _resetCache();
});

describe('bookmarks', () => {
  describe('addBookmark', () => {
    it('adds a new bookmark', async () => {
      const bookmark = await addBookmark('Hello world', 'user', 'msg-1');
      expect(bookmark.content).toBe('Hello world');
      expect(bookmark.role).toBe('user');
      expect(bookmark.sessionId).toBe('msg-1');
      expect(bookmark.id).toMatch(/^bm-/);
      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        STORAGE_KEY,
        expect.stringContaining('Hello world'),
      );
    });

    it('deduplicates by content + sessionId', async () => {
      const existing = [{
        id: 'bm-1',
        content: 'Hello world',
        role: 'user',
        sessionId: 'msg-1',
        savedAt: 100,
      }];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(existing));

      const result = await addBookmark('Hello world', 'user', 'msg-1');
      expect(result.id).toBe('bm-1');
      expect(AsyncStorage.setItem).not.toHaveBeenCalled();
    });

    it('adds different content as new bookmark', async () => {
      const existing = [{
        id: 'bm-1',
        content: 'Hello world',
        role: 'user',
        sessionId: 'msg-1',
        savedAt: 100,
      }];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(existing));

      const result = await addBookmark('Different content', 'user', 'msg-1');
      expect(result.id).not.toBe('bm-1');
      expect(AsyncStorage.setItem).toHaveBeenCalled();
    });
  });

  describe('getBookmarks', () => {
    it('returns empty array when nothing stored', async () => {
      const result = await getBookmarks();
      expect(result).toEqual([]);
    });

    it('returns stored bookmarks', async () => {
      const bookmarks = [{
        id: 'bm-1',
        content: 'Test',
        role: 'assistant',
        sessionId: 'msg-1',
        savedAt: 100,
      }];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(bookmarks));

      const result = await getBookmarks();
      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('Test');
    });

    it('uses cache on subsequent calls', async () => {
      const bookmarks = [{
        id: 'bm-1',
        content: 'Cached',
        role: 'user',
        sessionId: 'msg-1',
        savedAt: 100,
      }];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(bookmarks));

      await getBookmarks();
      await getBookmarks();
      expect(AsyncStorage.getItem).toHaveBeenCalledTimes(1);
    });
  });

  describe('removeBookmark', () => {
    it('removes bookmark by id', async () => {
      const bookmarks = [
        {id: 'bm-1', content: 'Keep', role: 'user', sessionId: 'msg-1', savedAt: 100},
        {id: 'bm-2', content: 'Remove', role: 'user', sessionId: 'msg-2', savedAt: 200},
      ];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(bookmarks));

      await removeBookmark('bm-2');
      const saved = JSON.parse((AsyncStorage.setItem as jest.Mock).mock.calls[0][1]);
      expect(saved).toHaveLength(1);
      expect(saved[0].id).toBe('bm-1');
    });
  });

  describe('isBookmarked', () => {
    it('returns true when bookmarked', async () => {
      const bookmarks = [{
        id: 'bm-1',
        content: 'Hello',
        role: 'user',
        sessionId: 'msg-1',
        savedAt: 100,
      }];
      (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(bookmarks));

      const result = await isBookmarked('Hello', 'msg-1');
      expect(result).toBe(true);
    });

    it('returns false when not bookmarked', async () => {
      const result = await isBookmarked('Not saved', 'msg-99');
      expect(result).toBe(false);
    });
  });
});
