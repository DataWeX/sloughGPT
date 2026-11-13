jest.mock('../toast', () => ({
  toast: {warn: jest.fn(), error: jest.fn(), success: jest.fn(), info: jest.fn()},
}));

jest.mock('../api-client', () => ({
  api: {get: jest.fn(), post: jest.fn()},
}));

const {getAllCommands} = require('../chat-commands');

describe('chat-commands', () => {
  it('returns an array of commands', () => {
    const commands = getAllCommands();
    expect(Array.isArray(commands)).toBe(true);
    expect(commands.length).toBeGreaterThan(0);
  });

  it('each command has command, description, usage, execute', () => {
    const commands = getAllCommands();
    for (const cmd of commands) {
      expect(typeof cmd.command).toBe('string');
      expect(cmd.command.startsWith('/')).toBe(true);
      expect(typeof cmd.description).toBe('string');
      expect(typeof cmd.usage).toBe('string');
      expect(typeof cmd.execute).toBe('function');
    }
  });

  it('/help lists all other commands', async () => {
    const commands = getAllCommands();
    const helpCmd = commands.find((c: any) => c.command === '/help');
    expect(helpCmd).toBeDefined();

    let systemMessage = '';
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: (msg: string) => { systemMessage = msg; },
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    const result = await helpCmd.execute([], ctx);
    expect(result.handled).toBe(true);
    expect(systemMessage).toContain('/clear');
    expect(systemMessage).toContain('/temp');
    expect(systemMessage).toContain('/model');
  });

  it('/clear calls clearChat and shows toast', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/clear');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    const result = await cmd.execute([], ctx);
    expect(result.handled).toBe(true);
    expect(ctx.clearChat).toHaveBeenCalled();
    expect(ctx.showToast).toHaveBeenCalledWith('Chat cleared', 'success');
  });

  it('/temp validates numeric range', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/temp');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    await cmd.execute(['3.0'], ctx);
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /temp <0.0–2.0>', 'error');

    jest.mocked(ctx.showToast).mockClear();
    jest.mocked(ctx.setTemperature).mockClear();
    await cmd.execute(['0.6'], ctx);
    expect(ctx.setTemperature).toHaveBeenCalledWith(0.6);
    expect(ctx.showToast).toHaveBeenCalledWith('Temperature set to 0.6', 'success');
  });

  it('/temp requires an argument', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/temp');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    await cmd.execute([], ctx);
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /temp <0.0–2.0>', 'error');
  });

  it('/rename calls renameConversation', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/rename');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    const result = await cmd.execute(['my chat'], ctx);
    expect(result.handled).toBe(true);
    expect(ctx.renameConversation).toHaveBeenCalledWith('my chat');
    expect(ctx.showToast).toHaveBeenCalledWith('Renamed to "my chat"', 'success');
  });

  it('/model shows error when no args', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/model');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    await cmd.execute([], ctx);
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /model <name>', 'error');
  });

  it('/export calls exportChat', async () => {
    const commands = getAllCommands();
    const cmd = commands.find((c: any) => c.command === '/export');
    const ctx = {
      showToast: jest.fn(),
      clearChat: jest.fn(),
      setTemperature: jest.fn(),
      setModel: jest.fn(async () => {}),
      setSoul: jest.fn(async () => {}),
      exportChat: jest.fn(),
      attachFile: jest.fn(),
      searchKnowledge: jest.fn(async () => {}),
      navigateTo: jest.fn(),
      addSystemMessage: jest.fn(),
      sendMessage: jest.fn(async () => {}),
      archiveConversation: jest.fn(),
      renameConversation: jest.fn(),
      searchConversations: jest.fn(),
    };

    await cmd.execute([], ctx);
    expect(ctx.exportChat).toHaveBeenCalled();
    expect(ctx.showToast).toHaveBeenCalledWith('Chat exported', 'success');
  });
});
