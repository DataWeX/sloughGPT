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

  it('calls onCheckedChange when toggled', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Switch onCheckedChange={onChange} />)
    await user.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledTimes(1)
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
