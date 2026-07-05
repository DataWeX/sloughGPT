import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

import { KeyboardShortcutsModal, ShortcutsHint, useKeyboardShortcuts } from './KeyboardShortcutsModal'

describe('KeyboardShortcutsModal', () => {
  it('renders dialog with title', () => {
    render(<KeyboardShortcutsModal open />)
    expect(screen.getAllByText('Keyboard Shortcuts').length >= 1).toBe(true)
  })

  it('renders shortcut categories', () => {
    render(<KeyboardShortcutsModal open />)
    expect(screen.getAllByText('Chat').length >= 1).toBe(true)
    expect(screen.getAllByText('General').length >= 1).toBe(true)
    expect(screen.getAllByText('Navigation').length >= 1).toBe(true)
  })

  it('renders shortcut descriptions', () => {
    render(<KeyboardShortcutsModal open />)
    expect(screen.getAllByText('Send message').length >= 1).toBe(true)
    expect(screen.getAllByText('Show keyboard shortcuts').length >= 1).toBe(true)
  })
})

describe('ShortcutsHint', () => {
  it('renders ? badge button', () => {
    render(<ShortcutsHint />)
    expect(screen.getAllByTitle('Keyboard shortcuts').length >= 1).toBe(true)
  })

  it('opens modal on click', async () => {
    render(<ShortcutsHint />)
    const buttons = screen.getAllByTitle('Keyboard shortcuts')
    fireEvent.click(buttons[0])
    await waitFor(() => {
      expect(screen.getAllByText('Keyboard Shortcuts').length >= 1).toBe(true)
    })
  })
})

describe('useKeyboardShortcuts', () => {
  it('returns showModal and setShowModal', () => {
    function Test() {
      const { showModal, setShowModal } = useKeyboardShortcuts()
      return <div data-testid="val">{String(showModal)}<button onClick={() => setShowModal(true)}>open</button></div>
    }
    render(<Test />)
    expect(screen.getByTestId('val')).toHaveTextContent('false')
  })
})
