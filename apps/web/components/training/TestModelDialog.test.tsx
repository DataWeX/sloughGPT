import { render, fireEvent, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'

import { TestModelDialog } from './TestModelDialog'

describe('TestModelDialog', () => {
  afterEach(() => {
    cleanup()
    document.body.innerHTML = ''
  })

  const base = {
    open: true, prompt: '', result: null, loading: false,
    onClose: vi.fn(), onPromptChange: vi.fn(), onGenerate: vi.fn(), onClear: vi.fn(),
  }

  it('returns null when not open', () => {
    render(<TestModelDialog {...base} open={false} />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('renders when open', () => {
    render(<TestModelDialog {...base} />)
    expect(screen.getByText('Test the model')).toBeInTheDocument()
  })

  it('renders textarea inside modal', () => {
    render(<TestModelDialog {...base} />)
    const textareas = document.body.querySelectorAll('textarea')
    expect(textareas.length).toBeGreaterThanOrEqual(1)
  })

  it('textarea value reflects prompt prop', () => {
    render(<TestModelDialog {...base} prompt="hello" />)
    const textarea = document.body.querySelector('textarea') as HTMLTextAreaElement
    expect(textarea).toHaveValue('hello')
  })

  it('calls onPromptChange on textarea input', () => {
    render(<TestModelDialog {...base} />)
    const textarea = document.body.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'hi' } })
    expect(base.onPromptChange).toHaveBeenCalledWith('hi')
  })

  it('Generate button disabled when prompt empty', () => {
    render(<TestModelDialog {...base} prompt="" />)
    const genBtn = [...document.body.querySelectorAll('button')].find(b => b.textContent === 'Generate')
    expect(genBtn).toBeDisabled()
  })

  it('Generate button enabled when prompt non-empty', () => {
    render(<TestModelDialog {...base} prompt="hello" />)
    const genBtn = [...document.body.querySelectorAll('button')].find(b => b.textContent === 'Generate')
    expect(genBtn).not.toBeDisabled()
  })

  it('Generate button shows Generating... when loading', () => {
    render(<TestModelDialog {...base} prompt="hello" loading />)
    expect(screen.getByText('Generating...')).toBeInTheDocument()
  })

  it('calls onGenerate on Generate click', () => {
    render(<TestModelDialog {...base} prompt="hello" />)
    const genBtn = [...document.body.querySelectorAll('button')].find(b => b.textContent === 'Generate')
    fireEvent.click(genBtn!)
    expect(base.onGenerate).toHaveBeenCalled()
  })

  it('calls onClear on Clear click', () => {
    render(<TestModelDialog {...base} />)
    const clearBtn = [...document.body.querySelectorAll('button')].find(b => b.textContent === 'Clear')
    fireEvent.click(clearBtn!)
    expect(base.onClear).toHaveBeenCalled()
  })

  it('shows output section when result has response', () => {
    const result = { prompt: 'hi', response: 'generated text', model: 'gpt2', tokens_generated: 3, error: '' }
    render(<TestModelDialog {...base} result={result} />)
    expect(screen.getByText('generated text')).toBeInTheDocument()
    expect(screen.getByText('Output')).toBeInTheDocument()
  })

  it('does not show output section when result is null', () => {
    render(<TestModelDialog {...base} result={null} />)
    expect(screen.queryByText('Output')).not.toBeInTheDocument()
  })

  it('shows error section when result has error', () => {
    const result = { prompt: 'hi', response: '', model: '', tokens_generated: 0, error: 'model not loaded' }
    render(<TestModelDialog {...base} result={result} />)
    expect(screen.getByText('model not loaded')).toBeInTheDocument()
    expect(screen.getByText('Error')).toBeInTheDocument()
  })

  it('shows model and token info', () => {
    const result = { prompt: 'hi', response: 'hello', model: 'gpt2', tokens_generated: 5, error: '' }
    render(<TestModelDialog {...base} result={result} />)
    expect(screen.getByText('Model: gpt2')).toBeInTheDocument()
    expect(screen.getByText('Tokens: 5')).toBeInTheDocument()
  })

  it('calls onClose on close button click', () => {
    render(<TestModelDialog {...base} />)
    const closeBtn = document.body.querySelector('button[aria-label="Close"]') as HTMLButtonElement
    expect(closeBtn).toBeInTheDocument()
    fireEvent.click(closeBtn)
    expect(base.onClose).toHaveBeenCalled()
  })
})
