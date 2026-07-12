import {describe, it, expect, vi, beforeEach} from 'vitest'
import {getAllCommands, type CommandContext} from './chat-commands'

function makeCtx(overrides?: Partial<CommandContext>): CommandContext {
  return {
    showToast: vi.fn(),
    clearChat: vi.fn(),
    setTemperature: vi.fn(),
    setModel: vi.fn().mockResolvedValue(undefined),
    setSoul: vi.fn().mockResolvedValue(undefined),
    exportChat: vi.fn(),
    attachFile: vi.fn(),
    searchKnowledge: vi.fn().mockResolvedValue(undefined),
    navigateTo: vi.fn(),
    addSystemMessage: vi.fn(),
    sendMessage: vi.fn().mockResolvedValue(undefined),
    archiveConversation: vi.fn(),
    renameConversation: vi.fn(),
    searchConversations: vi.fn(),
    ...overrides,
  }
}

describe('chat-commands', () => {
  const cmds = getAllCommands()

  it('returns an array of commands', () => {
    expect(cmds.length).toBeGreaterThan(0)
  })

  it('each command has required fields', () => {
    for (const cmd of cmds) {
      expect(cmd.command).toBeTruthy()
      expect(cmd.description).toBeTruthy()
      expect(cmd.usage).toBeTruthy()
      expect(typeof cmd.execute).toBe('function')
    }
  })

  // /help
  it('/help lists commands', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/help')!
    const result = await cmd.execute([], ctx)
    expect(result.handled).toBe(true)
    expect(ctx.addSystemMessage).toHaveBeenCalled()
  })

  // /clear
  it('/clear clears chat', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/clear')!
    await cmd.execute([], ctx)
    expect(ctx.clearChat).toHaveBeenCalled()
    expect(ctx.showToast).toHaveBeenCalledWith('Chat cleared', 'success')
  })

  // /temp
  it('/temp sets temperature', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/temp')!
    await cmd.execute(['0.8'], ctx)
    expect(ctx.setTemperature).toHaveBeenCalledWith(0.8)
  })

  it('/temp rejects invalid values', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/temp')!
    await cmd.execute(['abc'], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /temp <0.0–2.0>', 'error')
    await cmd.execute(['3.0'], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /temp <0.0–2.0>', 'error')
    await cmd.execute(['-1'], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith('Usage: /temp <0.0–2.0>', 'error')
  })

  // /model
  it('/model switches model', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/model')!
    await cmd.execute(['gpt2'], ctx)
    expect(ctx.setModel).toHaveBeenCalledWith('gpt2')
    expect(ctx.addSystemMessage).toHaveBeenCalled()
  })

  it('/model with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/model')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/model'), 'error')
  })

  it('/model handles error', async () => {
    const ctx = makeCtx({setModel: vi.fn().mockRejectedValue(new Error('fail'))})
    const cmd = cmds.find(c => c.command === '/model')!
    await cmd.execute(['bad-model'], ctx)
    const lastMsg = (ctx.addSystemMessage as any).mock.calls.at(-1)?.[0]
    expect(lastMsg).toContain('Failed')
  })

  // /soul
  it('/soul switches soul', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/soul')!
    await cmd.execute(['friendly'], ctx)
    expect(ctx.setSoul).toHaveBeenCalledWith('friendly')
  })

  it('/soul with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/soul')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/soul'), 'error')
  })

  it('/soul handles error', async () => {
    const ctx = makeCtx({setSoul: vi.fn().mockRejectedValue(new Error('not found'))})
    const cmd = cmds.find(c => c.command === '/soul')!
    await cmd.execute(['bad-soul'], ctx)
    const lastMsg = (ctx.addSystemMessage as any).mock.calls.at(-1)?.[0]
    expect(lastMsg).toContain('not found')
  })

  // /export
  it('/export exports chat', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/export')!
    await cmd.execute([], ctx)
    expect(ctx.exportChat).toHaveBeenCalled()
  })

  // /file
  it('/file attaches file', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/file')!
    await cmd.execute([], ctx)
    expect(ctx.attachFile).toHaveBeenCalled()
  })

  // /knowledge
  it('/knowledge searches knowledge base', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/knowledge')!
    await cmd.execute(['python async'], ctx)
    expect(ctx.searchKnowledge).toHaveBeenCalledWith('python async')
  })

  it('/knowledge with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/knowledge')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/knowledge'), 'error')
  })

  // /goto
  it('/goto navigates', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/goto')!
    await cmd.execute(['/models'], ctx)
    expect(ctx.navigateTo).toHaveBeenCalledWith('/models')
  })

  it('/goto with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/goto')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/goto'), 'error')
  })

  // /summarize
  it('/summarize sends message', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/summarize')!
    await cmd.execute([], ctx)
    expect(ctx.addSystemMessage).toHaveBeenCalled()
    expect(ctx.sendMessage).toHaveBeenCalled()
  })

  // /feedback
  it('/feedback records feedback', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/feedback')!
    await cmd.execute(['positive', 'great answer'], ctx)
    expect(ctx.addSystemMessage).toHaveBeenCalled()
    expect(ctx.showToast).toHaveBeenCalledWith('Feedback recorded', 'success')
  })

  it('/feedback with invalid sentiment shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/feedback')!
    await cmd.execute(['maybe'], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/feedback'), 'error')
  })

  it('/feedback with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/feedback')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/feedback'), 'error')
  })

  // /translate
  it('/translate sends translation request', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/translate')!
    await cmd.execute(['Spanish'], ctx)
    expect(ctx.addSystemMessage).toHaveBeenCalled()
    expect(ctx.sendMessage).toHaveBeenCalled()
  })

  it('/translate with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/translate')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/translate'), 'error')
  })

  // /search
  it('/search searches conversations', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/search')!
    await cmd.execute(['python'], ctx)
    expect(ctx.searchConversations).toHaveBeenCalledWith('python')
  })

  it('/search with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/search')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/search'), 'error')
  })

  // /archive
  it('/archive archives conversation', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/archive')!
    await cmd.execute([], ctx)
    expect(ctx.archiveConversation).toHaveBeenCalled()
  })

  // /rename
  it('/rename renames conversation', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/rename')!
    await cmd.execute(['My Chat'], ctx)
    expect(ctx.renameConversation).toHaveBeenCalledWith('My Chat')
  })

  it('/rename with no args shows usage', async () => {
    const ctx = makeCtx()
    const cmd = cmds.find(c => c.command === '/rename')!
    await cmd.execute([], ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('/rename'), 'error')
  })
})
