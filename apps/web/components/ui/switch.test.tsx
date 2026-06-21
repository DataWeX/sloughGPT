/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Switch } from './switch'

afterEach(cleanup)

describe('Switch', () => {
  it('renders with switch role', () => {
    render(<Switch />)
    expect(screen.getByRole('switch')).toBeInTheDocument()
  })

  it('is unchecked by default', () => {
    render(<Switch />)
    const el = screen.getByRole('switch')
    expect(el).toHaveAttribute('data-state', 'unchecked')
  })

  it('reflects checked state when defaultChecked', () => {
    render(<Switch defaultChecked />)
    const el = screen.getByRole('switch')
    expect(el).toHaveAttribute('data-state', 'checked')
  })

  it('calls onCheckedChange when toggled', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Switch onCheckedChange={onChange} />)
    await user.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('calls onCheckedChange with false when toggled off', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Switch defaultChecked onCheckedChange={onChange} />)
    await user.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(false)
  })

  it('is disabled when disabled prop is set', () => {
    render(<Switch disabled />)
    expect(screen.getByRole('switch')).toBeDisabled()
  })

  it('applies custom className', () => {
    const { container } = render(<Switch className="custom-class" />)
    const btn = container.querySelector('button')
    expect(btn).toHaveClass('custom-class')
  })

  it('renders thumb element', () => {
    const { container } = render(<Switch />)
    const thumb = container.querySelector('.rounded-full')
    expect(thumb).toBeInTheDocument()
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Switch ref={ref} />)
    expect(ref).toHaveBeenCalled()
  })
})
