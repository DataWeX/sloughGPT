import { describe, expect, it, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Checkbox } from './checkbox'

describe('Checkbox', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders a checkbox input with role checkbox', () => {
    render(<Checkbox />)
    const input = screen.getByRole('checkbox')
    expect(input.getAttribute('type')).toBe('checkbox')
  })

  it('defaults to unchecked', () => {
    render(<Checkbox />)
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false)
  })

  it('reflects defaultChecked', () => {
    render(<Checkbox defaultChecked />)
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(true)
  })

  it('toggles on click and fires onCheckedChange', () => {
    const onCheckedChange = vi.fn()
    render(<Checkbox onCheckedChange={onCheckedChange} />)
    const input = screen.getByRole('checkbox') as HTMLInputElement
    fireEvent.click(input)
    expect(onCheckedChange).toHaveBeenCalledWith(true)
    expect(input.checked).toBe(true)
    fireEvent.click(input)
    expect(onCheckedChange).toHaveBeenCalledWith(false)
    expect(input.checked).toBe(false)
  })

  it('fires onChange alongside onCheckedChange', () => {
    const onChange = vi.fn()
    const onCheckedChange = vi.fn()
    render(<Checkbox onChange={onChange} onCheckedChange={onCheckedChange} />)
    fireEvent.click(screen.getByRole('checkbox'))
    expect(onChange).toHaveBeenCalled()
    expect(onCheckedChange).toHaveBeenCalledWith(true)
  })

  it('is controlled by the checked prop', () => {
    const onCheckedChange = vi.fn()
    const { rerender } = render(<Checkbox checked={true} onCheckedChange={onCheckedChange} />)
    const input = screen.getByRole('checkbox') as HTMLInputElement
    expect(input.checked).toBe(true)
    fireEvent.click(input)
    expect(onCheckedChange).toHaveBeenCalledWith(false)
    rerender(<Checkbox checked={false} onCheckedChange={onCheckedChange} />)
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false)
  })

  it('renders the input with the disabled attribute', () => {
    const onCheckedChange = vi.fn()
    render(<Checkbox disabled onCheckedChange={onCheckedChange} />)
    const input = screen.getByRole('checkbox') as HTMLInputElement
    expect(input.disabled).toBe(true)
  })

  it('does not render aria-checked attribute but exposes state via checked', () => {
    render(<Checkbox defaultChecked />)
    const input = screen.getByRole('checkbox') as HTMLInputElement
    expect(input.checked).toBe(true)
  })

  it('renders label and description', () => {
    render(<Checkbox id="terms" label="Accept terms" description="By continuing you agree to the terms" />)
    expect(screen.getByText('Accept terms')).toBeTruthy()
    expect(screen.getByText('By continuing you agree to the terms')).toBeTruthy()
    const label = screen.getByText('Accept terms').closest('label')
    expect(label?.getAttribute('for')).toBe('terms')
  })

  it('sets the indeterminate property', () => {
    render(<Checkbox indeterminate />)
    expect((screen.getByRole('checkbox') as HTMLInputElement).indeterminate).toBe(true)
  })

  it('applies size classes to wrapper', () => {
    const { container } = render(<Checkbox size="sm" />)
    const wrapper = container.querySelector('span')
    expect(wrapper?.className).toContain('h-4 w-4')
  })

  it('shows checkmark SVG when checked', () => {
    const { container } = render(<Checkbox defaultChecked />)
    const svg = container.querySelector('svg')
    expect(svg).toBeTruthy()
  })

  it('shows indeterminate dash when indeterminate', () => {
    const { container } = render(<Checkbox indeterminate />)
    const svgs = container.querySelectorAll('svg')
    expect(svgs.length).toBeGreaterThanOrEqual(2)
  })
})
