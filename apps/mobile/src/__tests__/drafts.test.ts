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

const {getDraft, saveDraft, clearDraft} = require('../services/drafts');

describe('drafts', () => {
  it('CRUD lifecycle', async () => {
    expect(await getDraft('s1')).toBe('');

    await saveDraft('s1', 'hello');
    expect(await getDraft('s1')).toBe('hello');

    await saveDraft('s1', 'updated');
    expect(await getDraft('s1')).toBe('updated');

    await saveDraft('s2', 'other');
    expect(await getDraft('s2')).toBe('other');

    await clearDraft('s1');
    expect(await getDraft('s1')).toBe('');
    expect(await getDraft('s2')).toBe('other');

    await saveDraft('s3', 'text');
    await saveDraft('s3', '  ');
    expect(await getDraft('s3')).toBe('');
  });
});
