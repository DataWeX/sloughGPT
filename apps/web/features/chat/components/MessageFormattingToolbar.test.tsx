import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { MessageFormattingToolbar } from './MessageFormattingToolbar'

afterEach(cleanup)

describe('MessageFormattingToolbar', () => {
  it('renders toolbar buttons', () => {
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={vi.fn()}
      />
    )
    expect(screen.getByText('More')).toBeInTheDocument()
    expect(screen.getByText('Copy')).toBeInTheDocument()
  })

  it('expands on More click', () => {
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    expect(screen.getByText('Less')).toBeInTheDocument()
    expect(screen.getByText('Bold')).toBeInTheDocument()
    expect(screen.getByText('Italic')).toBeInTheDocument()
    expect(screen.getByText('Code')).toBeInTheDocument()
  })

  it('calls onFormat with bold', () => {
    const onFormat = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={onFormat}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    fireEvent.click(screen.getByText('Bold'))
    expect(onFormat).toHaveBeenCalledWith('**hello**')
  })

  it('calls onFormat with italic', () => {
    const onFormat = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={onFormat}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    fireEvent.click(screen.getByText('Italic'))
    expect(onFormat).toHaveBeenCalledWith('_hello_')
  })

  it('calls onFormat with code', () => {
    const onFormat = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={onFormat}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    fireEvent.click(screen.getByText('Code'))
    expect(onFormat).toHaveBeenCalledWith('`hello`')
  })

  it('copies text', async () => {
    const onCopy = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={onCopy}
      />
    )
    await act(async () => {
      fireEvent.click(screen.getByText('Copy'))
    })
    expect(onCopy).toHaveBeenCalledWith('hello')
    expect(screen.getByText('Copied!')).toBeInTheDocument()
  })

  it('resets copied state after 2 seconds', async () => {
    vi.useFakeTimers()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={vi.fn()}
      />
    )
    await act(async () => {
      fireEvent.click(screen.getByText('Copy'))
    })
    await act(async () => {
      vi.advanceTimersByTime(2000)
    })
    expect(screen.getByText('Copy')).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('shows clear button when text provided', () => {
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={vi.fn()}
        onClear={vi.fn()}
      />
    )
    expect(screen.getByText('Clear')).toBeInTheDocument()
  })

  it('hides clear button when no text', () => {
    render(
      <MessageFormattingToolbar
        text=""
        onFormat={vi.fn()}
        onCopy={vi.fn()}
        onClear={vi.fn()}
      />
    )
    expect(screen.queryByText('Clear')).not.toBeInTheDocument()
  })

  it('calls onClear when clear clicked', () => {
    const onClear = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={vi.fn()}
        onCopy={vi.fn()}
        onClear={onClear}
      />
    )
    fireEvent.click(screen.getByText('Clear'))
    expect(onClear).toHaveBeenCalled()
  })

  it('formats list', () => {
    const onFormat = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={onFormat}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    fireEvent.click(screen.getByText('List'))
    expect(onFormat).toHaveBeenCalledWith('\n- hello')
  })

  it('formats quote', () => {
    const onFormat = vi.fn()
    render(
      <MessageFormattingToolbar
        text="hello"
        onFormat={onFormat}
        onCopy={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('More'))
    fireEvent.click(screen.getByText('Quote'))
    expect(onFormat).toHaveBeenCalledWith('\n> hello')
  })
})