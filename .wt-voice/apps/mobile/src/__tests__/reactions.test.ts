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

const {
  getMessageReactions,
  toggleReaction,
  getAllReactions,
  REACTION_EMOJIS,
} = require('../services/reactions');

describe('reactions', () => {
  it('CRUD lifecycle', async () => {
    // empty initially
    expect(await getMessageReactions('msg-1')).toEqual([]);
    expect(await getAllReactions()).toEqual({});

    // add reaction
    const r1 = await toggleReaction('msg-1', '👍');
    expect(r1).toContain('👍');

    // add another
    const r2 = await toggleReaction('msg-1', '❤️');
    expect(r2).toContain('👍');
    expect(r2).toContain('❤️');

    // toggle off
    const r3 = await toggleReaction('msg-1', '👍');
    expect(r3).not.toContain('👍');
    expect(r3).toContain('❤️');

    // multiple messages
    await toggleReaction('msg-2', '🔥');
    const all = await getAllReactions();
    expect(Object.keys(all)).toHaveLength(2);

    // REACTION_EMOJIS is exported
    expect(REACTION_EMOJIS.length).toBeGreaterThan(0);
  });
});
