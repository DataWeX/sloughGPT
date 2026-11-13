import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatInputAgentBox } from './ChatInputAgentBox'

afterEach(cleanup)

describe('ChatInputAgentBox', () => {
  it('renders textarea', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByPlaceholderText(/Type a message/)).toBeInTheDocument()
  })

  it('renders context indicator', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        contextTokens={50000}
        maxContext={128000}
      />
    )
    expect(screen.getByText(/Context:/)).toBeInTheDocument()
  })

  it('shows token count when typing', () => {
    render(
      <ChatInputAgentBox
        value="Hello world"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByText('Tokens:')).toBeInTheDocument()
  })

  it('shows word count', () => {
    render(
      <ChatInputAgentBox
        value="Hello world test"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByText('Words:')).toBeInTheDocument()
  })

  it('shows char count', () => {
    render(
      <ChatInputAgentBox
        value="Hello"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByText('Chars:')).toBeInTheDocument()
  })

  it('shows line count', () => {
    render(
      <ChatInputAgentBox
        value="Line 1\nLine 2"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByText('Lines:')).toBeInTheDocument()
  })

  it('shows estimated cost', () => {
    render(
      <ChatInputAgentBox
        value="Hello world"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        model="gpt-4"
      />
    )
    expect(screen.getByText('Est. cost:')).toBeInTheDocument()
  })

  it('shows model badge', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        model="gpt-4-turbo"
      />
    )
    expect(screen.getByText('gpt-4-turbo')).toBeInTheDocument()
  })

  it('shows tier badge', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        tier="pro"
      />
    )
    expect(screen.getByText('pro')).toBeInTheDocument()
  })

  it('opens quick actions', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText(/Quick/))
    expect(screen.getByText('Explain')).toBeInTheDocument()
    expect(screen.getByText('Summarize')).toBeInTheDocument()
  })

  it('filters actions by category', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText(/Quick/))
    fireEvent.click(screen.getByText('format'))
    expect(screen.getByText('Bold')).toBeInTheDocument()
    expect(screen.getByText('Italic')).toBeInTheDocument()
    expect(screen.queryByText('Explain')).not.toBeInTheDocument()
  })

  it('calls onInsertAction when action clicked', () => {
    const onInsertAction = vi.fn()
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={onInsertAction}
      />
    )
    fireEvent.click(screen.getByText(/Quick/))
    fireEvent.click(screen.getByText('Explain'))
    expect(onInsertAction).toHaveBeenCalledWith('Explain this concept in simple terms:')
  })

  it('toggles stats visibility', () => {
    render(
      <ChatInputAgentBox
        value="Hello"
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    expect(screen.getByText('Tokens:')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Hide Stats'))
    expect(screen.queryByText('Tokens:')).not.toBeInTheDocument()
  })

  it('shows context warning', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        contextTokens={110000}
        maxContext={128000}
      />
    )
    expect(screen.getByText(/used/)).toBeInTheDocument()
  })

  it('shows context critical', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
        contextTokens={125000}
        maxContext={128000}
      />
    )
    expect(screen.getByText(/used/)).toBeInTheDocument()
  })

  it('calls onChange when typing', () => {
    const onChange = vi.fn()
    render(
      <ChatInputAgentBox
        value=""
        onChange={onChange}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    fireEvent.change(screen.getByPlaceholderText(/Type a message/), {
      target: { value: 'Hello' },
    })
    expect(onChange).toHaveBeenCalledWith('Hello')
  })

  it('calls onSend on Ctrl+Enter', () => {
    const onSend = vi.fn()
    render(
      <ChatInputAgentBox
        value="Hello"
        onChange={vi.fn()}
        onSend={onSend}
        onInsertAction={vi.fn()}
      />
    )
    fireEvent.keyDown(screen.getByPlaceholderText(/Type a message/), {
      key: 'Enter',
      ctrlKey: true,
    })
    expect(onSend).toHaveBeenCalled()
  })

  it('closes quick actions on Escape', () => {
    render(
      <ChatInputAgentBox
        value=""
        onChange={vi.fn()}
        onSend={vi.fn()}
        onInsertAction={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText(/Quick/))
    expect(screen.getByText('Explain')).toBeInTheDocument()
    fireEvent.keyDown(screen.getByPlaceholderText(/Type a message/), { key: 'Escape' })
    expect(screen.queryByText('Explain')).not.toBeInTheDocument()
  })
})