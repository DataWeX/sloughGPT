import AsyncStorage from '@react-native-async-storage/async-storage';

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

beforeEach(async () => {
  await AsyncStorage.clear();
  jest.resetModules();
});

function getModule() {
  return require('../quick-prompts');
}

describe('quick-prompts', () => {
  it('getQuickPrompts returns defaults on first run', async () => {
    const qp = getModule();
    const prompts = await qp.getQuickPrompts();
    expect(Array.isArray(prompts)).toBe(true);
    expect(prompts.length).toBe(6);
    expect(prompts[0].title).toBe('Explain concept');
  });

  it('addQuickPrompt returns new prompt with custom id', async () => {
    const qp = getModule();
    const added = await qp.addQuickPrompt('Custom', 'Do {x}');
    expect(added.id).toContain('custom-');
    expect(added.title).toBe('Custom');
    expect(added.prompt).toBe('Do {x}');
    expect(added.category).toBe('custom');
  });

  it('deleteQuickPrompt removes a prompt', async () => {
    const qp = getModule();
    const added = await qp.addQuickPrompt('Delete me', 'text');
    await qp.deleteQuickPrompt(added.id);
    const all = await qp.getQuickPrompts();
    expect(all.find((p: any) => p.id === added.id)).toBeUndefined();
  });

  it('updateQuickPrompt modifies existing', async () => {
    const qp = getModule();
    const added = await qp.addQuickPrompt('Old title', 'old prompt');
    await qp.updateQuickPrompt(added.id, {title: 'New title'});
    const all = await qp.getQuickPrompts();
    expect(all.find((p: any) => p.id === added.id).title).toBe('New title');
  });

  it('getQuickPromptsByCategory filters', async () => {
    const qp = getModule();
    await qp.addQuickPrompt('Code prompt', 'debug this', 'coding');
    const coding = await qp.getQuickPromptsByCategory('coding');
    expect(coding.every((p: any) => p.category === 'coding')).toBe(true);
  });

  it('getQuickPromptsByCategory "all" returns all', async () => {
    const qp = getModule();
    const all = await qp.getQuickPromptsByCategory('all');
    const total = await qp.getQuickPrompts();
    expect(all.length).toBe(total.length);
  });

  it('fillTemplate replaces placeholders', () => {
    const qp = getModule();
    expect(qp.fillTemplate('Hello {name}', {name: 'World'})).toBe('Hello World');
  });

  it('fillTemplate without params returns raw', () => {
    const qp = getModule();
    expect(qp.fillTemplate('No placeholders')).toBe('No placeholders');
  });

  it('fillTemplate leaves unknown keys as-is', () => {
    const qp = getModule();
    expect(qp.fillTemplate('{unknown}')).toBe('{unknown}');
  });
});
