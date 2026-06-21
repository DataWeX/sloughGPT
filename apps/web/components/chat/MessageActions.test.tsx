/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { MessageActions } from './MessageActions'

afterEach(cleanup)

beforeEach(() => {
  Object.assign(navigator, {
    clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
  })
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
    expect(container.querySelectorAll('button')).toHaveLength(0)
  })

  it('renders all handlers simultaneously', () => {
    const { container } = render(
      <MessageActions
        content="text"
        messageId="m1"
        onCopy={vi.fn()}
        onRegenerate={vi.fn()}
        onThumbsUp={vi.fn()}
        onThumbsDown={vi.fn()}
        onEdit={vi.fn()}
      />
    )
    expect(container.querySelector('button[aria-label="Copy message"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Regenerate response"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Mark as helpful"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Mark as unhelpful"]')).toBeInTheDocument()
    expect(container.querySelector('button[aria-label="Edit and resend message"]')).toBeInTheDocument()
  })
})
