// @vitest-environment jsdom
import { render, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

import { TestModelDialog } from './TestModelDialog'

describe('TestModelDialog', () => {
  const base = {
    open: true, prompt: '', output: '', loading: false,
    onClose: vi.fn(), onPromptChange: vi.fn(), onGenerate: vi.fn(), onClear: vi.fn(),
  }

  it('returns null when not open', () => {
    const { container } = render(<TestModelDialog {...base} open={false} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders when open', () => {
    const { container } = render(<TestModelDialog {...base} />)
    expect(container.textContent).toContain('Test the model')
  })

  it('renders textarea inside modal', () => {
    const { container } = render(<TestModelDialog {...base} />)
    expect(container.querySelector('textarea')).toBeInTheDocument()
  })

  it('textarea value reflects prompt prop', () => {
    const { container } = render(<TestModelDialog {...base} prompt="hello" />)
    expect(container.querySelector('textarea')).toHaveValue('hello')
  })

  it('calls onPromptChange on textarea input', () => {
    const { container } = render(<TestModelDialog {...base} />)
    fireEvent.change(container.querySelector('textarea')!, { target: { value: 'hi' } })
    expect(base.onPromptChange).toHaveBeenCalledWith('hi')
  })

  it('Generate button disabled when prompt empty', () => {
    const { container } = render(<TestModelDialog {...base} prompt="" />)
    const btns = [...container.querySelectorAll('button')]
    const genBtn = btns.find(b => b.textContent === 'Generate')
    expect(genBtn).toBeDisabled()
  })

  it('Generate button enabled when prompt non-empty', () => {
    const { container } = render(<TestModelDialog {...base} prompt="hello" />)
    const genBtn = [...container.querySelectorAll('button')].find(b => b.textContent === 'Generate')
    expect(genBtn).not.toBeDisabled()
  })

  it('Generate button shows Generating... when loading', () => {
    const { container } = render(<TestModelDialog {...base} prompt="hello" loading />)
    expect(container.textContent).toContain('Generating...')
  })

  it('calls onGenerate on Generate click', () => {
    const { container } = render(<TestModelDialog {...base} prompt="hello" />)
    const genBtn = [...container.querySelectorAll('button')].find(b => b.textContent === 'Generate')
    fireEvent.click(genBtn!)
    expect(base.onGenerate).toHaveBeenCalled()
  })

  it('calls onClear on Clear click', () => {
    const { container } = render(<TestModelDialog {...base} />)
    const clearBtn = [...container.querySelectorAll('button')].find(b => b.textContent === 'Clear')
    fireEvent.click(clearBtn!)
    expect(base.onClear).toHaveBeenCalled()
  })

  it('shows output section when output present', () => {
    const { container } = render(<TestModelDialog {...base} output="generated text" />)
    expect(container.textContent).toContain('generated text')
    expect(container.textContent).toContain('Output')
  })

  it('does not show output section when output empty', () => {
    const { container } = render(<TestModelDialog {...base} output="" />)
    expect(container.textContent).not.toContain('Output')
  })

  it('calls onClose on backdrop click', () => {
    const { container } = render(<TestModelDialog {...base} />)
    fireEvent.click(container.firstChild as HTMLElement)
    expect(base.onClose).toHaveBeenCalled()
  })
})
