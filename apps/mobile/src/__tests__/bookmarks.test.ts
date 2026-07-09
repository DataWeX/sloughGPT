const store: Record<string, string> = {};

jest.mock('@react-native-async-storage/async-storage', () => ({
  __esModule: true,
  default: {
    getItem: jest.fn((key: string) => Promise.resolve(store[key] || null)),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
      return Promise.resolve();
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
      return Promise.resolve();
    }),
    clear: jest.fn(() => {
      Object.keys(store).forEach(k => delete store[k]);
      return Promise.resolve();
    }),
  },
}));

const {getBookmarks, addBookmark, removeBookmark, isBookmarked} =
  require('../services/bookmarks');

describe('bookmarks', () => {
  beforeEach(() => {
    Object.keys(store).forEach(k => delete store[k]);
  });

  it('CRUD lifecycle', async () => {
    // starts empty
    expect(await getBookmarks()).toEqual([]);
    expect(await isBookmarked('hello', 'msg-1')).toBe(false);

    // add
    const bm = await addBookmark('hello', 'user', 'msg-1');
    expect(bm.content).toBe('hello');
    expect(bm.role).toBe('user');

    // retrieve
    expect(await getBookmarks()).toHaveLength(1);
    expect(await isBookmarked('hello', 'msg-1')).toBe(true);

    // dedup
    await addBookmark('hello', 'user', 'msg-1');
    expect(await getBookmarks()).toHaveLength(1);

    // different sessionId
    await addBookmark('hello', 'user', 'msg-2');
    expect(await getBookmarks()).toHaveLength(2);

    // remove
    await removeBookmark(bm.id);
    expect(await getBookmarks()).toHaveLength(1);
    expect(await isBookmarked('hello', 'msg-1')).toBe(false);
  });
});
