import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/chat-commands', () => ({
  getAllCommands: () => [
    { command: '/help', description: 'Show available commands', usage: '/help', execute: vi.fn() },
    { command: '/clear', description: 'Clear the chat history', usage: '/clear', execute: vi.fn() },
    { command: '/model', description: 'Switch the active model', usage: '/model <name>', execute: vi.fn() },
    { command: '/temp', description: 'Set the temperature (0.0 – 2.0)', usage: '/temp <value>', execute: vi.fn() },
  ],
}))

import { SlashCommandMenu } from './SlashCommandMenu'

describe('SlashCommandMenu', () => {
  const onInsert = vi.fn()
  const onClose = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders all commands when value is just "/"', () => {
    render(<SlashCommandMenu value="/" onInsert={onInsert} onClose={onClose} />)
    expect(screen.getByRole('listbox')).toBeDefined()
    expect(screen.getByText('/help')).toBeDefined()
    expect(screen.getByText('/clear')).toBeDefined()
    expect(screen.getByText('/model')).toBeDefined()
    expect(screen.getByText('/temp')).toBeDefined()
  })

  it('filters commands by query', () => {
    render(<SlashCommandMenu value="/mo" onInsert={onInsert} onClose={onClose} />)
    expect(screen.getByText('/model')).toBeDefined()
    expect(screen.queryByText('/help')).toBeNull()
  })

  it('calls onInsert and onClose on click', () => {
    render(<SlashCommandMenu value="/" onInsert={onInsert} onClose={onClose} />)
    fireEvent.click(screen.getByText('/help'))
    expect(onInsert).toHaveBeenCalledWith('/help')
    expect(onClose).toHaveBeenCalled()
  })

  it('supports keyboard navigation with ArrowDown and Enter', () => {
    render(<SlashCommandMenu value="/" onInsert={onInsert} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onInsert).toHaveBeenCalled()
  })

  it('closes on Escape', () => {
    render(<SlashCommandMenu value="/" onInsert={onInsert} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('returns null when no commands match', () => {
    const { container } = render(<SlashCommandMenu value="/zzzzz" onInsert={onInsert} onClose={onClose} />)
    expect(container.innerHTML).toBe('')
  })

  describe('onExecute', () => {
    const onExecute = vi.fn()

    it('calls onExecute with command and args on click', () => {
      render(<SlashCommandMenu value="/temp 0.5" onInsert={onInsert} onClose={onClose} onExecute={onExecute} />)
      fireEvent.click(screen.getByText('/temp'))
      expect(onExecute).toHaveBeenCalledWith(
        expect.objectContaining({ command: '/temp' }),
        ['0.5'],
      )
      expect(onClose).toHaveBeenCalled()
    })

    it('parses multiple args from value', () => {
      render(<SlashCommandMenu value="/model gpt2 large" onInsert={onInsert} onClose={onClose} onExecute={onExecute} />)
      fireEvent.click(screen.getByText('/model'))
      expect(onExecute).toHaveBeenCalledWith(
        expect.objectContaining({ command: '/model' }),
        ['gpt2', 'large'],
      )
    })

    it('passes empty args when no extra text', () => {
      render(<SlashCommandMenu value="/clear" onInsert={onInsert} onClose={onClose} onExecute={onExecute} />)
      fireEvent.click(screen.getByText('/clear'))
      expect(onExecute).toHaveBeenCalledWith(
        expect.objectContaining({ command: '/clear' }),
        [],
      )
    })

    it('calls onExecute on Enter when onExecute is set', () => {
      render(<SlashCommandMenu value="/help" onInsert={onInsert} onClose={onClose} onExecute={onExecute} />)
      fireEvent.keyDown(window, { key: 'Enter' })
      expect(onExecute).toHaveBeenCalled()
      expect(onInsert).not.toHaveBeenCalled()
    })

    it('falls back to onInsert when onExecute is not provided', () => {
      render(<SlashCommandMenu value="/help" onInsert={onInsert} onClose={onClose} />)
      fireEvent.click(screen.getByText('/help'))
      expect(onInsert).toHaveBeenCalledWith('/help')
    })
  })
})
