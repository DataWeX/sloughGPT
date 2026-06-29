import { describe, it, expect, vi } from 'vitest'
import { getCommand, getAllCommands } from './chat-commands'

describe('getCommand', () => {
  it('returns null for normal messages', () => {
    expect(getCommand('hello world')).toBeNull()
    expect(getCommand('')).toBeNull()
    expect(getCommand('  ')).toBeNull()
  })

  it('returns null for unknown commands', () => {
    expect(getCommand('/foobar')).toBeNull()
    expect(getCommand('/xyz abc')).toBeNull()
  })

  it('recognizes /help', () => {
    const result = getCommand('/help')
    expect(result).not.toBeNull()
    expect(result!.command.command).toBe('/help')
    expect(result!.args).toEqual([])
  })

  it('parses arguments after command', () => {
    const result = getCommand('/model gpt2')
    expect(result).not.toBeNull()
    expect(result!.command.command).toBe('/model')
    expect(result!.args).toEqual(['gpt2'])
  })

  it('handles multiple arguments', () => {
    const result = getCommand('/temp 0.8')
    expect(result).not.toBeNull()
    expect(result!.command.command).toBe('/temp')
    expect(result!.args).toEqual(['0.8'])
  })

  it('is case insensitive', () => {
    expect(getCommand('/CLEAR')).not.toBeNull()
    expect(getCommand('/Help')).not.toBeNull()
  })
})

describe('getAllCommands', () => {
  it('returns all commands', () => {
    const cmds = getAllCommands()
    expect(cmds.length).toBeGreaterThanOrEqual(11)
    const names = cmds.map(c => c.command)
    expect(names).toContain('/help')
    expect(names).toContain('/clear')
    expect(names).toContain('/model')
    expect(names).toContain('/temp')
  })
})

describe('/help command', () => {
  it('returns a help message listing all commands', async () => {
    const cmd = getCommand('/help')!
    const ctx = {
      showToast: vi.fn(),
      clearChat: vi.fn(),
      setTemperature: vi.fn(),
      setModel: vi.fn(),
      setSoul: vi.fn(),
      exportChat: vi.fn(),
      attachFile: vi.fn(),
      searchKnowledge: vi.fn(),
      navigateTo: vi.fn(),
      addSystemMessage: vi.fn(),
      sendMessage: vi.fn(),
    }
    const result = await cmd.command.execute(cmd.args, ctx)
    expect(result.handled).toBe(true)
    expect(ctx.addSystemMessage).toHaveBeenCalledOnce()
    const msg = ctx.addSystemMessage.mock.calls[0][0] as string
    expect(msg).toContain('Chat Commands')
    const names = getAllCommands().map(c => c.command)
    for (const n of names) {
      if (n !== '/help') expect(msg).toContain(n)
    }
  })
})

describe('/temp command', () => {
  it('shows error for invalid temperature', async () => {
    const cmd = getCommand('/temp abc')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(),
      sendMessage: vi.fn(),
    }
    await cmd.command.execute(cmd.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
    expect(ctx.setTemperature).not.toHaveBeenCalled()
  })

  it('sets temperature for valid value', async () => {
    const cmd = getCommand('/temp 0.5')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(),
      sendMessage: vi.fn(),
    }
    await cmd.command.execute(cmd.args, ctx)
    expect(ctx.setTemperature).toHaveBeenCalledWith(0.5)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('0.5'), 'success')
  })
})

describe('/soul command', () => {
  it('shows error with no args', async () => {
    const cmd = getCommand('/soul')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn().mockRejectedValue(new Error('not found')),
      exportChat: vi.fn(), attachFile: vi.fn(), searchKnowledge: vi.fn(),
      navigateTo: vi.fn(), addSystemMessage: vi.fn(), sendMessage: vi.fn(),
    }
    await cmd.command.execute(cmd.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
  })
})

describe('/knowledge command', () => {
  it('shows error with no args', async () => {
    const cmd = getCommand('/knowledge')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(),
      sendMessage: vi.fn(),
    }
    await cmd.command.execute(cmd.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
  })
})

describe('/summarize command', () => {
  it('injects a summarisation system message', async () => {
    const cmd = getCommand('/summarize')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(),
      sendMessage: vi.fn(),
    }
    const result = await cmd!.command.execute(cmd!.args, ctx)
    expect(result.handled).toBe(true)
    expect(ctx.addSystemMessage).toHaveBeenCalledOnce()
    const msg = ctx.addSystemMessage.mock.calls[0][0] as string
    expect(msg).toContain('summarise')
    expect(msg).toContain('bullet points')
    expect(ctx.sendMessage).toHaveBeenCalledOnce()
  })
})

describe('/feedback command', () => {
  const makeCtx = () => ({
    showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
    setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
    attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
    addSystemMessage: vi.fn(),
    sendMessage: vi.fn(),
  })

  it('shows error with no args', async () => {
    const cmd = getCommand('/feedback')!
    const ctx = makeCtx()
    await cmd!.command.execute(cmd!.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
  })

  it('shows error with invalid sentiment', async () => {
    const cmd = getCommand('/feedback neutral')!
    const ctx = makeCtx()
    await cmd!.command.execute(cmd!.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
  })

  it('records positive feedback without reason', async () => {
    const cmd = getCommand('/feedback positive')!
    const ctx = makeCtx()
    const result = await cmd!.command.execute(cmd!.args, ctx)
    expect(result.handled).toBe(true)
    expect(ctx.addSystemMessage).toHaveBeenCalledWith(expect.stringContaining('positive'))
    expect(ctx.showToast).toHaveBeenCalledWith('Feedback recorded', 'success')
  })

  it('records negative feedback with reason', async () => {
    const cmd = getCommand('/feedback negative too verbose')!
    const ctx = makeCtx()
    const result = await cmd!.command.execute(cmd!.args, ctx)
    expect(result.handled).toBe(true)
    const msg = ctx.addSystemMessage.mock.calls[0][0] as string
    expect(msg).toContain('negative')
    expect(msg).toContain('too verbose')
    expect(ctx.showToast).toHaveBeenCalledWith('Feedback recorded', 'success')
  })
})

describe('/translate command', () => {
  it('shows error with no args', async () => {
    const cmd = getCommand('/translate')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(), sendMessage: vi.fn(),
    }
    await cmd!.command.execute(cmd!.args, ctx)
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('Usage'), 'error')
  })

  it('adds a system message and triggers generation', async () => {
    const cmd = getCommand('/translate Spanish')!
    const ctx = {
      showToast: vi.fn(), clearChat: vi.fn(), setTemperature: vi.fn(),
      setModel: vi.fn(), setSoul: vi.fn(), exportChat: vi.fn(),
      attachFile: vi.fn(), searchKnowledge: vi.fn(), navigateTo: vi.fn(),
      addSystemMessage: vi.fn(), sendMessage: vi.fn(),
    }
    const result = await cmd!.command.execute(cmd!.args, ctx)
    expect(result.handled).toBe(true)
    expect(ctx.addSystemMessage).toHaveBeenCalledWith(expect.stringContaining('Spanish'))
    expect(ctx.sendMessage).toHaveBeenCalledOnce()
  })
})
