/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react'
import { MessageActions } from './MessageActions'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  delete (navigator as any).clipboard
})

beforeEach(() => {
  // navigator.clipboard is a read-only getter in jsdom — define an own
  // configurable property so tests can stub writeText.
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
  })
  // stub speechSynthesis + SpeechSynthesisUtterance for jsdom
  if (!('speechSynthesis' in window)) {
    const mockUtterance = vi.fn()
    ;(window as any).SpeechSynthesisUtterance = mockUtterance
    ;(window as any).speechSynthesis = {
      speak: vi.fn(),
      cancel: vi.fn(),
      speaking: false,
      paused: false,
      pending: false,
    }
  }
})

describe('MessageActions', () => {
  it('renders copy button when onCopy provided', () => {
    render(<MessageActions content="hello" messageId="m1" onCopy={vi.fn()} />)
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument()
  })

  it('does not render copy button when onCopy not provided', () => {
    render(<MessageActions content="hello" messageId="m1" />)
    expect(screen.queryByRole('button', { name: /copy/i })).not.toBeInTheDocument()
  })

  it('renders regenerate button when onRegenerate provided', () => {
    render(<MessageActions content="hello" messageId="m1" onRegenerate={vi.fn()} />)
    expect(screen.getByRole('button', { name: /regenerate/i })).toBeInTheDocument()
  })

  it('does not render regenerate button without handler', () => {
    render(<MessageActions content="hello" messageId="m1" />)
    expect(screen.queryByRole('button', { name: /regenerate/i })).not.toBeInTheDocument()
  })

  it('renders thumbs up button when onThumbsUp provided', () => {
    render(<MessageActions content="hello" messageId="m1" onThumbsUp={vi.fn()} />)
    expect(screen.getByRole('button', { name: /helpful/i })).toBeInTheDocument()
  })

  it('renders thumbs down button when onThumbsDown provided', () => {
    render(<MessageActions content="hello" messageId="m1" onThumbsDown={vi.fn()} />)
    expect(screen.getByRole('button', { name: /unhelpful/i })).toBeInTheDocument()
  })

  it('renders edit button when onEdit provided', () => {
    render(<MessageActions content="hello" messageId="m1" onEdit={vi.fn()} />)
    expect(screen.getByRole('button', { name: /edit/i })).toBeInTheDocument()
  })

  it('calls onCopy with content when copy clicked', async () => {
    const onCopy = vi.fn()
    render(<MessageActions content="copy this" messageId="m1" onCopy={onCopy} />)
    fireEvent.click(screen.getByRole('button', { name: /copy message/i }))
    await vi.waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('copy this')
      expect(onCopy).toHaveBeenCalledWith('copy this')
    })
  })

  it('shows check icon after copy', async () => {
    render(<MessageActions content="text" messageId="m1" onCopy={vi.fn()} />)
    const copyBtn = screen.getByRole('button', { name: /copy message/i })
    fireEvent.click(copyBtn)
    await vi.waitFor(() => {
      expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
    })
  })

  it('calls onRegenerate when regenerate clicked', () => {
    const onRegen = vi.fn()
    render(<MessageActions content="text" messageId="m1" onRegenerate={onRegen} />)
    fireEvent.click(screen.getByRole('button', { name: /regenerate/i }))
    expect(onRegen).toHaveBeenCalledTimes(1)
  })

  it('calls onThumbsUp when thumbs up clicked', async () => {
    const onUp = vi.fn()
    render(<MessageActions content="text" messageId="m1" onThumbsUp={onUp} />)
    fireEvent.click(screen.getByRole('button', { name: 'Mark as helpful' }))
    await vi.waitFor(() => {
      expect(onUp).toHaveBeenCalledWith('m1')
    })
  })

  it('toggles thumbs up on double click', async () => {
    const onUp = vi.fn()
    render(<MessageActions content="text" messageId="m1" onThumbsUp={onUp} />)
    const btn = screen.getByRole('button', { name: 'Mark as helpful' })
    fireEvent.click(btn)
    fireEvent.click(btn)
    await vi.waitFor(() => {
      expect(onUp).toHaveBeenCalledTimes(2)
    })
  })

  it('clears thumbs down when thumbs up is clicked', async () => {
    const onUp = vi.fn()
    const onDown = vi.fn()
    render(<MessageActions content="text" messageId="m1" onThumbsUp={onUp} onThumbsDown={onDown} />)
    const downBtn = screen.getByRole('button', { name: 'Mark as unhelpful' })
    fireEvent.click(downBtn)
    await vi.waitFor(() => {
      expect(downBtn).toHaveAttribute('aria-pressed', 'true')
    })
    const upBtn = screen.getByRole('button', { name: 'Mark as helpful' })
    fireEvent.click(upBtn)
    await vi.waitFor(() => {
      expect(upBtn).toHaveAttribute('aria-pressed', 'true')
      expect(downBtn).toHaveAttribute('aria-pressed', 'false')
    })
  })

  it('calls onThumbsDown when thumbs down clicked', async () => {
    const onDown = vi.fn()
    render(<MessageActions content="text" messageId="m1" onThumbsDown={onDown} />)
    fireEvent.click(screen.getByRole('button', { name: 'Mark as unhelpful' }))
    await vi.waitFor(() => {
      expect(onDown).toHaveBeenCalledWith('m1')
    })
  })

  it('calls onEdit when edit clicked', () => {
    const onEdit = vi.fn()
    render(<MessageActions content="text" messageId="m1" onEdit={onEdit} />)
    fireEvent.click(screen.getByRole('button', { name: /edit/i }))
    expect(onEdit).toHaveBeenCalledWith('m1')
  })

  it('has role group with accessible label', () => {
    render(<MessageActions content="text" messageId="m1" onCopy={vi.fn()} />)
    const group = screen.getByRole('group')
    expect(group).toHaveAttribute('aria-label', 'Message actions')
  })

  it('does not render any buttons with no handlers', () => {
    const { container } = render(<MessageActions content="text" messageId="m1" />)
    const btns = container.querySelectorAll('button')
    const nonReactionBtns = Array.from(btns).filter(b => b.getAttribute('aria-label') !== 'Add reaction')
    expect(nonReactionBtns).toHaveLength(0)
  })

  it('renders all handlers simultaneously', () => {
    const { container } = render(
      <MessageActions
        content="text"
        messageId="m1"
        role="assistant"
        onCopy={vi.fn()}
        onRegenerate={vi.fn()}
        onThumbsUp={vi.fn()}
        onThumbsDown={vi.fn()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />
    )
    expect(container.querySelector('button[aria-label="Copy message"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Regenerate response"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Mark as helpful"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Mark as unhelpful"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Edit and resend message"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Delete message"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Read aloud"]')).toBeInTheDocument()
  })

  it('renders delete button when onDelete provided', () => {
    render(<MessageActions content="text" messageId="m1" onDelete={vi.fn()} />)
    expect(screen.getByRole('button', { name: /delete message/i })).toBeInTheDocument()
  })

  it('does not render delete button without handler', () => {
    render(<MessageActions content="text" messageId="m1" />)
    expect(screen.queryByRole('button', { name: /delete message/i })).not.toBeInTheDocument()
  })

  it('calls onDelete with messageId when delete clicked', () => {
    const onDelete = vi.fn()
    render(<MessageActions content="text" messageId="m1" onDelete={onDelete} />)
    fireEvent.click(screen.getByRole('button', { name: /delete message/i }))
    expect(onDelete).toHaveBeenCalledWith('m1')
  })

  it('renders read aloud button for assistant role', () => {
    render(<MessageActions content="hello" messageId="m1" role="assistant" />)
    expect(screen.getByRole('button', { name: /read aloud/i })).toBeInTheDocument()
  })

  it('does not render read aloud button without role', () => {
    render(<MessageActions content="hello" messageId="m1" />)
    expect(screen.queryByRole('button', { name: /read aloud/i })).not.toBeInTheDocument()
  })

  it('does not render read aloud button for user role', () => {
    render(<MessageActions content="hello" messageId="m1" role="user" />)
    expect(screen.queryByRole('button', { name: /read aloud/i })).not.toBeInTheDocument()
  })

  it('calls speechSynthesis.speak when read aloud clicked', () => {
    const speak = vi.fn()
    window.speechSynthesis.speak = speak
    render(<MessageActions content="hello world" messageId="m1" role="assistant" />)
    fireEvent.click(screen.getByRole('button', { name: /read aloud/i }))
    expect(speak).toHaveBeenCalledTimes(1)
    // SpeechSynthesisUtterance should have been constructed with the content
    expect(window.SpeechSynthesisUtterance).toHaveBeenCalledWith('hello world')
  })

  it('cancels speech when clicked again', () => {
    const cancel = vi.fn()
    window.speechSynthesis.cancel = cancel
    render(<MessageActions content="hello" messageId="m1" role="assistant" />)
    const btn = screen.getByRole('button', { name: /read aloud/i })
    fireEvent.click(btn)
    // after first click, speaking is true — click again to cancel
    fireEvent.click(btn)
    expect(cancel).toHaveBeenCalledTimes(1)
  })

  it('shows stop icon when speaking', () => {
    render(<MessageActions content="hello" messageId="m1" role="assistant" />)
    const btn = screen.getByRole('button', { name: /read aloud/i })
    fireEvent.click(btn)
    // after click, speaking is true, label should change
    expect(screen.getByRole('button', { name: /stop reading aloud/i })).toBeInTheDocument()
  })
})

// Add a test for when speechSynthesis is not available
describe('MessageActions without speechSynthesis', () => {
  beforeEach(() => {
    delete (window as any).speechSynthesis
    delete (window as any).SpeechSynthesisUtterance
  })

  afterEach(() => {
    cleanup()
  })

  it('does not render read aloud button when speechSynthesis is unavailable', () => {
    render(<MessageActions content="hello" messageId="m1" role="assistant" />)
    expect(screen.queryByRole('button', { name: /read aloud/i })).not.toBeInTheDocument()
  })
})
