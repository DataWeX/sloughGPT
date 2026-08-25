import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ContextInjectionBar } from './ContextInjectionBar'

afterEach(cleanup)

describe('ContextInjectionBar', () => {
  it('renders button to show input', () => {
    render(<ContextInjectionBar onInject={vi.fn()} />)
    expect(screen.getByText('+ Context')).toBeInTheDocument()
  })

  it('shows input when button clicked', () => {
    render(<ContextInjectionBar onInject={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Context'))
    expect(screen.getByPlaceholderText('Add context...')).toBeInTheDocument()
  })

  it('calls onInject when Enter pressed', () => {
    const onInject = vi.fn()
    render(<ContextInjectionBar onInject={onInject} />)
    fireEvent.click(screen.getByText('+ Context'))
    const input = screen.getByPlaceholderText('Add context...')
    fireEvent.change(input, { target: { value: 'test context' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onInject).toHaveBeenCalledWith('test context')
  })

  it('calls onInject when Send clicked', () => {
    const onInject = vi.fn()
    render(<ContextInjectionBar onInject={onInject} />)
    fireEvent.click(screen.getByText('+ Context'))
    const input = screen.getByPlaceholderText('Add context...')
    fireEvent.change(input, { target: { value: 'test context' } })
    fireEvent.click(screen.getByText('Send'))
    expect(onInject).toHaveBeenCalledWith('test context')
  })

  it('clears input and hides on Escape', () => {
    render(<ContextInjectionBar onInject={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Context'))
    const input = screen.getByPlaceholderText('Add context...')
    fireEvent.change(input, { target: { value: 'test' } })
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Add context...')).not.toBeInTheDocument()
  })

  it('hides input after injection', () => {
    render(<ContextInjectionBar onInject={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Context'))
    const input = screen.getByPlaceholderText('Add context...')
    fireEvent.change(input, { target: { value: 'test context' } })
    fireEvent.click(screen.getByText('Send'))
    expect(screen.queryByPlaceholderText('Add context...')).not.toBeInTheDocument()
  })

  it('disables button when disabled prop is true', () => {
    render(<ContextInjectionBar onInject={vi.fn()} disabled />)
    const button = screen.getByText('+ Context')
    expect(button).toBeDisabled()
  })
})
