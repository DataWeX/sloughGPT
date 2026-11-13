import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

beforeEach(() => {
  AsyncStorage.clear();
  jest.resetModules();
});

const drafts = require('../drafts');

describe('drafts', () => {
  it('getDraft returns empty string for unknown session', async () => {
    expect(await drafts.getDraft('sess1')).toBe('');
  });

  it('saveDraft stores text', async () => {
    await drafts.saveDraft('sess1', 'hello');
    expect(await drafts.getDraft('sess1')).toBe('hello');
  });

  it('saveDraft with empty text removes draft', async () => {
    await drafts.saveDraft('sess1', 'hello');
    await drafts.saveDraft('sess1', '  ');
    expect(await drafts.getDraft('sess1')).toBe('');
  });

  it('clearDraft removes a draft', async () => {
    await drafts.saveDraft('sess1', 'hi');
    await drafts.clearDraft('sess1');
    expect(await drafts.getDraft('sess1')).toBe('');
  });

  it('isolates drafts by session', async () => {
    await drafts.saveDraft('a', 'aaa');
    await drafts.saveDraft('b', 'bbb');
    expect(await drafts.getDraft('a')).toBe('aaa');
    expect(await drafts.getDraft('b')).toBe('bbb');
    await drafts.clearDraft('a');
    expect(await drafts.getDraft('b')).toBe('bbb');
  });
});
