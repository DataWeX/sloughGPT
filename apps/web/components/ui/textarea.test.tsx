/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Textarea } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Textarea', () => {
  it('renders as textarea element', () => {
    render(<Textarea />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })

  it('applies placeholder', () => {
    render(<Textarea placeholder="Type here..." />)
    expect(screen.getByPlaceholderText('Type here...')).toBeInTheDocument()
  })

  it('sets value', () => {
    render(<Textarea value="content" onChange={() => {}} />)
    const el = screen.getByRole('textbox') as HTMLTextAreaElement
    expect(el.value).toBe('content')
  })

  it('calls onChange when typing', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Textarea onChange={onChange} />)
    await user.type(screen.getByRole('textbox'), 'a')
    expect(onChange).toHaveBeenCalled()
  })

  it('is disabled when disabled prop is set', () => {
    render(<Textarea disabled />)
    expect(screen.getByRole('textbox')).toBeDisabled()
  })

  it('applies custom className', () => {
    const { container } = render(<Textarea className="my-class" />)
    expect(container.firstChild).toHaveClass('my-class')
  })

  it('has min-h-20 and resize-y classes', () => {
    const { container } = render(<Textarea />)
    expect(container.firstChild).toHaveClass('min-h-20')
    expect(container.firstChild).toHaveClass('resize-y')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Textarea ref={ref} />)
    expect(ref).toHaveBeenCalledOnce()
  })
})
