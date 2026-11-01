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

const qp = require('../services/quick-prompts');

describe('quick-prompts', () => {
  it('CRUD + template fill', async () => {
    // starts with defaults
    const all = await qp.getQuickPrompts();
    expect(all.length).toBeGreaterThan(0);

    // category filter
    const coding = await qp.getQuickPromptsByCategory('coding');
    expect(coding.every((p: any) => p.category === 'coding')).toBe(true);

    const allFiltered = await qp.getQuickPromptsByCategory('all');
    expect(allFiltered.length).toBe(all.length);

    // add custom
    const custom = await qp.addQuickPrompt('Test', 'Say {topic}', 'custom');
    expect(custom.title).toBe('Test');
    expect(custom.id).toMatch(/^custom-/);

    // update
    await qp.updateQuickPrompt(custom.id, {title: 'Updated'});
    const updated = (await qp.getQuickPrompts()).find((p: any) => p.id === custom.id);
    expect(updated.title).toBe('Updated');

    // delete
    await qp.deleteQuickPrompt(custom.id);
    const afterDelete = await qp.getQuickPrompts();
    expect(afterDelete.find((p: any) => p.id === custom.id)).toBeUndefined();

    // fillTemplate
    expect(qp.fillTemplate('Hello {name}', {name: 'World'})).toBe('Hello World');
    expect(qp.fillTemplate('No params')).toBe('No params');
    expect(qp.fillTemplate('{a} {b}', {a: 'X'})).toBe('X {b}');
  });
});
