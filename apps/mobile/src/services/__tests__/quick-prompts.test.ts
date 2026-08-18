import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

function fresh() {
  jest.resetModules();
  return require('../quick-prompts');
}

beforeEach(() => {
  AsyncStorage.clear();
});

describe('quick-prompts', () => {
  it('getQuickPrompts returns defaults on first run', async () => {
    const qp = fresh();
    const prompts = await qp.getQuickPrompts();
    expect(Array.isArray(prompts)).toBe(true);
    expect(prompts.length).toBeGreaterThanOrEqual(6);
    expect(prompts[0].title).toBe('Explain concept');
  });

  it('caches after first load', async () => {
    const qp = fresh();
    await qp.getQuickPrompts();
    const raw = await AsyncStorage.getItem('@sloughgpt/quick-prompts');
    expect(raw).toBeTruthy();
  });

  it('addQuickPrompt appends a new prompt', async () => {
    const qp = fresh();
    const before = await qp.getQuickPrompts();
    const added = await qp.addQuickPrompt('Custom', 'Do {x}');
    expect(added.id).toContain('custom-');
    const after = await qp.getQuickPrompts();
    expect(after.length).toBe(before.length + 1);
  });

  it('deleteQuickPrompt removes a prompt', async () => {
    const qp = fresh();
    const added = await qp.addQuickPrompt('Delete me', 'text');
    await qp.deleteQuickPrompt(added.id);
    const all = await qp.getQuickPrompts();
    expect(all.find((p: any) => p.id === added.id)).toBeUndefined();
  });

  it('updateQuickPrompt modifies existing', async () => {
    const qp = fresh();
    const added = await qp.addQuickPrompt('Old title', 'old prompt');
    await qp.updateQuickPrompt(added.id, {title: 'New title'});
    const all = await qp.getQuickPrompts();
    expect(all.find((p: any) => p.id === added.id).title).toBe('New title');
  });

  it('getQuickPromptsByCategory filters', async () => {
    const qp = fresh();
    await qp.addQuickPrompt('Code prompt', 'debug this', 'coding');
    const coding = await qp.getQuickPromptsByCategory('coding');
    expect(coding.every((p: any) => p.category === 'coding')).toBe(true);
  });

  it('getQuickPromptsByCategory "all" returns all', async () => {
    const qp = fresh();
    const all = await qp.getQuickPromptsByCategory('all');
    const total = await qp.getQuickPrompts();
    expect(all.length).toBe(total.length);
  });

  it('fillTemplate replaces placeholders', () => {
    const qp = fresh();
    const result = qp.fillTemplate('Hello {name}', {name: 'World'});
    expect(result).toBe('Hello World');
  });

  it('fillTemplate without params returns raw', () => {
    const qp = fresh();
    expect(qp.fillTemplate('No placeholders')).toBe('No placeholders');
  });

  it('fillTemplate leaves unknown keys as-is', () => {
    const qp = fresh();
    expect(qp.fillTemplate('{unknown}')).toBe('{unknown}');
  });
});
