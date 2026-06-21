/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Checkbox } from './checkbox'

afterEach(cleanup)

describe('Checkbox', () => {
  it('renders as checkbox input', () => {
    render(<Checkbox />)
    const el = screen.getByRole('checkbox') as HTMLInputElement
    expect(el.type).toBe('checkbox')
  })

  it('is unchecked by default', () => {
    render(<Checkbox />)
    const el = screen.getByRole('checkbox') as HTMLInputElement
    expect(el.checked).toBe(false)
  })

  it('reflects checked prop', () => {
    render(<Checkbox checked onChange={() => {}} />)
    const el = screen.getByRole('checkbox') as HTMLInputElement
    expect(el.checked).toBe(true)
  })

  it('calls onChange on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Checkbox onChange={onChange} />)
    await user.click(screen.getByRole('checkbox'))
    expect(onChange).toHaveBeenCalledOnce()
  })

  it('calls onCheckedChange with checked state', async () => {
    const onCheckedChange = vi.fn()
    const user = userEvent.setup()
    render(<Checkbox onCheckedChange={onCheckedChange} />)
    await user.click(screen.getByRole('checkbox'))
    expect(onCheckedChange).toHaveBeenCalledWith(true)
  })

  it('is disabled when disabled prop is set', () => {
    render(<Checkbox disabled />)
    expect(screen.getByRole('checkbox')).toBeDisabled()
  })

  it('applies custom className', () => {
    const { container } = render(<Checkbox className="custom-checkbox" />)
    expect(container.firstChild).toHaveClass('custom-checkbox')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Checkbox ref={ref} />)
    expect(ref).toHaveBeenCalledOnce()
  })
})
