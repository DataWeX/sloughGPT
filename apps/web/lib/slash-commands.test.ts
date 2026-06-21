import { describe, expect, it } from 'vitest'
import { slashCommands, findMatchingCommands, parseSlashCommand } from './slash-commands'

describe('slashCommands', () => {
  it('has all 10 commands', () => {
    expect(slashCommands).toHaveLength(10)
  })

  it('each command has required fields', () => {
    for (const cmd of slashCommands) {
      expect(cmd.name).toBeTruthy()
      expect(cmd.description).toBeTruthy()
      expect(typeof cmd.prompt).toBe('function')
    }
  })

  it('prompt functions produce text with input', () => {
    const summarize = slashCommands.find(c => c.name === 'summarize')!
    expect(summarize.prompt('long text')).toContain('long text')
    expect(summarize.prompt('long text')).toContain('Summarize')
  })

  it('code command requests code generation', () => {
    const code = slashCommands.find(c => c.name === 'code')!
    expect(code.prompt('build a web app')).toContain('build a web app')
    expect(code.prompt('build a web app')).toContain('Generate clean')
  })
})

describe('findMatchingCommands', () => {
  it('returns all commands for empty query', () => {
    expect(findMatchingCommands('')).toHaveLength(10)
  })

  it('returns all commands for whitespace', () => {
    expect(findMatchingCommands('  ')).toHaveLength(10)
  })

  it('filters by name prefix', () => {
    const r = findMatchingCommands('sum')
    expect(r).toHaveLength(1)
    expect(r[0].name).toBe('summarize')
  })

  it('matches by description', () => {
    const r = findMatchingCommands('bullet')
    expect(r).toHaveLength(1)
    expect(r[0].name).toBe('bullet')
  })

  it('is case-insensitive', () => {
    expect(findMatchingCommands('DEBUG')).toHaveLength(1)
  })

  it('returns empty for no match', () => {
    expect(findMatchingCommands('zzzzz')).toEqual([])
  })
})

describe('parseSlashCommand', () => {
  it('parses /summarize with text', () => {
    const r = parseSlashCommand('/summarize this is the text')
    expect(r.command).not.toBeNull()
    expect(r.command!.name).toBe('summarize')
    expect(r.text).toBe('this is the text')
  })

  it('parses /code without trailing text', () => {
    const r = parseSlashCommand('/code')
    expect(r.command!.name).toBe('code')
    expect(r.text).toBe('')
  })

  it('returns null command and full text for non-slash input', () => {
    const r = parseSlashCommand('just normal text')
    expect(r.command).toBeNull()
    expect(r.text).toBe('just normal text')
  })

  it('returns null command for unknown slash command', () => {
    const r = parseSlashCommand('/nonexistent do stuff')
    expect(r.command).toBeNull()
    expect(r.text).toBe('do stuff')
  })

  it('trims trailing whitespace from text', () => {
    const r = parseSlashCommand('/translate   hello world   ')
    expect(r.command!.name).toBe('translate')
    expect(r.text).toBe('hello world')
  })

  it('handles multiline text after command', () => {
    const r = parseSlashCommand('/explain line1\nline2\nline3')
    expect(r.command!.name).toBe('explain')
    expect(r.text).toBe('line1\nline2\nline3')
  })

  it('slash without valid command name after it returns null', () => {
    const r = parseSlashCommand('/   ')  // slash followed by spaces
    expect(r.command).toBeNull()
    expect(r.text).toBe('/   ')
  })
})
