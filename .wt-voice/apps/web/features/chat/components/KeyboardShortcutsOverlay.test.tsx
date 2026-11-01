import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { KeyboardShortcutsOverlay } from './KeyboardShortcutsOverlay'

afterEach(cleanup)

describe('KeyboardShortcutsOverlay', () => {
  it('renders title', () => {
    render(<KeyboardShortcutsOverlay onClose={vi.fn()} />)
    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument()
  })

  it('renders all categories', () => {
    render(<KeyboardShortcutsOverlay onClose={vi.fn()} />)
    expect(screen.getByText('General')).toBeInTheDocument()
    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.getByText('Search')).toBeInTheDocument()
    expect(screen.getByText('Tools')).toBeInTheDocument()
  })

  it('renders shortcut keys', () => {
    render(<KeyboardShortcutsOverlay onClose={vi.fn()} />)
    expect(screen.getByText('New chat')).toBeInTheDocument()
    expect(screen.getByText('Regenerate response')).toBeInTheDocument()
    expect(screen.getByText('Focus search')).toBeInTheDocument()
    expect(screen.getByText('Approve tool call')).toBeInTheDocument()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<KeyboardShortcutsOverlay onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onClose).toHaveBeenCalled()
  })

  it('renders keyboard keys as kbd elements', () => {
    render(<KeyboardShortcutsOverlay onClose={vi.fn()} />)
    const kbdElements = screen.getAllByText('Ctrl')
    expect(kbdElements.length).toBeGreaterThan(0)
    expect(kbdElements[0].tagName).toBe('KBD')
  })
})