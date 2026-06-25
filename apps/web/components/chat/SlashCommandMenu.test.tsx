// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockCommands = [
  { name: 'help', description: 'Show help', icon: '📖', category: 'general', action: vi.fn() },
  { name: 'clear', description: 'Clear chat', icon: '🧹', category: 'general', action: vi.fn() },
  { name: 'summarize', description: 'Summarize conversation', icon: '📝', category: 'tools', action: vi.fn() },
]

vi.mock('@/lib/slash-commands', () => ({
  findMatchingCommands: (q: string) => {
    if (!q) return mockCommands
    return mockCommands.filter(c => c.name.includes(q.toLowerCase()))
  },
}))

import { SlashCommandMenu } from './SlashCommandMenu'

describe('SlashCommandMenu', () => {
  const onSelect = vi.fn()
  const onClose = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders nothing when not visible', () => {
    const { container } = render(
      <SlashCommandMenu query="" visible={false} onSelect={onSelect} onClose={onClose} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when visible but no matches', () => {
    const { container } = render(
      <SlashCommandMenu query="zzz_nonexistent" visible={true} onSelect={onSelect} onClose={onClose} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders matching commands', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    expect(screen.getByText('/help')).toBeDefined()
    expect(screen.getByText('/clear')).toBeDefined()
    expect(screen.getByText('/summarize')).toBeDefined()
  })

  it('filters commands by query', () => {
    render(<SlashCommandMenu query="help" visible={true} onSelect={onSelect} onClose={onClose} />)
    expect(screen.getByText('/help')).toBeDefined()
    expect(screen.queryByText('/clear')).toBeNull()
    expect(screen.queryByText('/summarize')).toBeNull()
  })

  it('calls onSelect when command clicked', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    fireEvent.click(screen.getByText('/help'))
    expect(onSelect).toHaveBeenCalledWith(mockCommands[0])
  })

  it('calls onClose when backdrop clicked', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    const backdrop = document.querySelector('.fixed.inset-0')
    fireEvent.click(backdrop!)
    expect(onClose).toHaveBeenCalled()
  })

  it('selects first command by default', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    const options = screen.getAllByRole('option')
    expect(options[0].getAttribute('aria-selected')).toBe('true')
  })

  it('navigates with arrow keys and selects with Enter', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onSelect).toHaveBeenCalledWith(mockCommands[1])
  })

  it('navigates up with ArrowUp', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'ArrowUp' })
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onSelect).toHaveBeenCalledWith(mockCommands[1])
  })

  it('closes on Escape', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('selects with Tab key', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'Tab' })
    expect(onSelect).toHaveBeenCalledWith(mockCommands[0])
  })

  it('updates selected index on mouse enter', () => {
    render(<SlashCommandMenu query="" visible={true} onSelect={onSelect} onClose={onClose} />)
    const options = screen.getAllByRole('option')
    fireEvent.mouseEnter(options[2])
    expect(options[2].getAttribute('aria-selected')).toBe('true')
  })

  it('resets selected index when query changes', () => {
    const { rerender } = render(<SlashCommandMenu query="help" visible={true} onSelect={onSelect} onClose={onClose} />)
    rerender(<SlashCommandMenu query="clear" visible={true} onSelect={onSelect} onClose={onClose} />)
    const options = screen.getAllByRole('option')
    expect(options[0].getAttribute('aria-selected')).toBe('true')
  })

  it('does not respond to keyboard when not visible', () => {
    const onSelectNotCalled = vi.fn()
    render(<SlashCommandMenu query="" visible={false} onSelect={onSelectNotCalled} onClose={onClose} />)
    fireEvent.keyDown(window, { key: 'Enter' })
    expect(onSelectNotCalled).not.toHaveBeenCalled()
  })
})
