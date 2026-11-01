import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

function fresh() {
  jest.resetModules();
  return require('../reactions');
}

beforeEach(() => {
  AsyncStorage.clear();
});

describe('reactions', () => {
  it('getMessageReactions returns empty for unknown', async () => {
    const r = fresh();
    expect(await r.getMessageReactions('msg1')).toEqual([]);
  });

  it('toggleReaction adds emoji', async () => {
    const r = fresh();
    const result = await r.toggleReaction('msg1', '👍');
    expect(result).toContain('👍');
  });

  it('toggleReaction removes existing emoji', async () => {
    const r = fresh();
    await r.toggleReaction('msg1', '👍');
    const result = await r.toggleReaction('msg1', '👍');
    expect(result).not.toContain('👍');
  });

  it('toggleReaction accumulates multiple emojis', async () => {
    const r = fresh();
    await r.toggleReaction('msg1', '❤️');
    await r.toggleReaction('msg1', '🔥');
    const result = await r.toggleReaction('msg1', '🎉');
    expect(result).toContain('❤️');
    expect(result).toContain('🔥');
    expect(result).toContain('🎉');
  });

  it('getAllReactions returns all', async () => {
    const r = fresh();
    await r.toggleReaction('a', '👍');
    await r.toggleReaction('b', '🔥');
    const all = await r.getAllReactions();
    expect(all.a).toContain('👍');
    expect(all.b).toContain('🔥');
  });

  it('exports REACTION_EMOJIS array', () => {
    const r = fresh();
    expect(r.REACTION_EMOJIS.length).toBe(10);
    expect(r.REACTION_EMOJIS).toContain('👍');
  });
});
