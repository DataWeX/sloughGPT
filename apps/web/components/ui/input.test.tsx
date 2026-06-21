/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Input, SearchInput } from './input'

afterEach(cleanup)

describe('Input', () => {
  it('renders as input element', () => {
    render(<Input />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })

  it('applies placeholder', () => {
    render(<Input placeholder="Enter name" />)
    expect(screen.getByPlaceholderText('Enter name')).toBeInTheDocument()
  })

  it('applies type from props', () => {
    render(<Input type="email" />)
    const el = screen.getByRole('textbox') as HTMLInputElement
    expect(el.type).toBe('email')
  })

  it('defaults to text type', () => {
    render(<Input />)
    const el = screen.getByRole('textbox') as HTMLInputElement
    expect(el.type).toBe('text')
  })

  it('sets value', () => {
    render(<Input value="hello" onChange={() => {}} />)
    const el = screen.getByRole('textbox') as HTMLInputElement
    expect(el.value).toBe('hello')
  })

  it('accepts className', () => {
    const { container } = render(<Input className="my-class" />)
    expect(container.firstChild).toHaveClass('my-class')
  })

  it('is disabled when disabled prop is set', () => {
    render(<Input disabled />)
    expect(screen.getByRole('textbox')).toBeDisabled()
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Input ref={ref} />)
    expect(ref).toHaveBeenCalledOnce()
  })

  it('has height class h-10', () => {
    const { container } = render(<Input />)
    expect(container.firstChild).toHaveClass('h-10')
  })
})

describe('SearchInput', () => {
  it('renders an input with type text', () => {
    render(<SearchInput />)
    const el = screen.getByRole('textbox') as HTMLInputElement
    expect(el.type).toBe('text')
  })

  it('renders search icon', () => {
    const { container } = render(<SearchInput />)
    const svg = container.querySelector('svg')
    expect(svg).toBeInTheDocument()
  })

  it('displays value', () => {
    render(<SearchInput value="search term" onChange={() => {}} />)
    const el = screen.getByRole('textbox') as HTMLInputElement
    expect(el.value).toBe('search term')
  })

  it('calls onChange with new value', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SearchInput value="" onChange={onChange} />)
    await user.type(screen.getByRole('textbox'), 'a')
    expect(onChange).toHaveBeenCalledWith('a')
  })

  it('applies placeholder', () => {
    render(<SearchInput placeholder="Search..." />)
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<SearchInput className="narrow" />)
    const input = container.querySelector('input')
    expect(input).toHaveClass('narrow')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<SearchInput ref={ref} />)
    expect(ref).toHaveBeenCalledOnce()
  })
})
