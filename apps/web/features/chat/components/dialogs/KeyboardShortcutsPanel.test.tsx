import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { KeyboardShortcutsPanel } from './KeyboardShortcutsPanel'

afterEach(cleanup)

describe('KeyboardShortcutsPanel', () => {
  it('returns null when closed', () => {
    const { container } = render(<KeyboardShortcutsPanel open={false} onClose={vi.fn()} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders title when open', () => {
    render(<KeyboardShortcutsPanel open={true} onClose={vi.fn()} />)
    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument()
  })

  it('renders all shortcuts', () => {
    render(<KeyboardShortcutsPanel open={true} onClose={vi.fn()} />)
    expect(screen.getByText('New chat')).toBeInTheDocument()
    expect(screen.getByText('Regenerate last response')).toBeInTheDocument()
    expect(screen.getAllByText('Focus search').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Cancel stream / close panel')).toBeInTheDocument()
    expect(screen.getByText('Toggle settings')).toBeInTheDocument()
  })

  it('calls onClose when clicking close button', () => {
    const onClose = vi.fn()
    render(<KeyboardShortcutsPanel open={true} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose when clicking backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(<KeyboardShortcutsPanel open={true} onClose={onClose} />)
    const backdrop = container.querySelector('.fixed')!
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalled()
  })

  it('does not propagate click on panel', () => {
    const onClose = vi.fn()
    render(<KeyboardShortcutsPanel open={true} onClose={onClose} />)
    const panel = screen.getByText('Keyboard Shortcuts').closest('.bg-background')!
    fireEvent.click(panel)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('formats Ctrl key as ⌘/Ctrl', () => {
    render(<KeyboardShortcutsPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText(/⌘\/Ctrl/).length).toBeGreaterThan(0)
  })

  it('formats Shift key as ⇧', () => {
    render(<KeyboardShortcutsPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('⇧').length).toBeGreaterThan(0)
  })
})
