import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi, afterEach } from 'vitest'

import { Slider, RangeSlider } from './slider'

afterEach(() => cleanup())

describe('Slider', () => {
  it('renders a range input with default props', () => {
    const { container } = render(<Slider />)
    const input = container.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(input).toBeTruthy()
    expect(input.value).toBe('0')
    expect(input.getAttribute('min')).toBe('0')
    expect(input.getAttribute('max')).toBe('100')
    expect(input.getAttribute('step')).toBe('1')
  })

  it('uses defaultValue for the initial value', () => {
    const { container } = render(<Slider defaultValue={[25]} />)
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.value).toBe('25')
  })

  it('renders a controlled value', () => {
    const { container } = render(<Slider value={[50]} />)
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.value).toBe('50')
  })

  it('calls onValueChange on change', () => {
    const onValueChange = vi.fn()
    const { container } = render(<Slider defaultValue={[25]} onValueChange={onValueChange} />)
    fireEvent.change(container.querySelector<HTMLInputElement>('input[type="range"]')!, { target: { value: '40' } })
    expect(onValueChange).toHaveBeenCalledWith([40])
  })

  it('updates the internal value when uncontrolled', () => {
    const { container } = render(<Slider defaultValue={[25]} />)
    fireEvent.change(container.querySelector<HTMLInputElement>('input[type="range"]')!, { target: { value: '40' } })
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.value).toBe('40')
  })

  it('keeps the controlled value after a change', () => {
    const { container } = render(<Slider value={[50]} />)
    fireEvent.change(container.querySelector<HTMLInputElement>('input[type="range"]')!, { target: { value: '70' } })
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.value).toBe('50')
  })

  it('sets disabled on the input', () => {
    const { container } = render(<Slider disabled />)
    const input = container.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(input.getAttribute('disabled')).not.toBeNull()
  })

  it('renders a label bound to the input id', () => {
    const { container } = render(<Slider id="vol" label="Volume" value={[40]} />)
    expect(screen.getByText('Volume')).toBeTruthy()
    const label = container.querySelector<HTMLLabelElement>('label')!
    expect(label.getAttribute('for')).toBe('vol')
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.getAttribute('id')).toBe('vol')
  })

  it('shows the current value when showValue is true', () => {
    render(<Slider value={[40]} showValue />)
    expect(screen.getByText('40')).toBeTruthy()
  })

  it('formats the displayed value with formatValue', () => {
    render(<Slider value={[40]} showValue formatValue={(v) => `${v}%`} />)
    expect(screen.getByText('40%')).toBeTruthy()
  })

  it('renders the fill width for the value', () => {
    const { container } = render(<Slider value={[50]} />)
    expect(container.querySelector<HTMLElement>('div.bg-primary')!.style.width).toBe('50%')
  })

  it('applies sm size classes', () => {
    const { container } = render(<Slider size="sm" />)
    const input = container.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(input.classList.contains('h-1')).toBe(true)
    expect(input.className).toContain('[&::-webkit-slider-thumb]:h-3')
  })

  it('applies default size classes', () => {
    const { container } = render(<Slider />)
    const input = container.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(input.classList.contains('h-2')).toBe(true)
    expect(input.className).toContain('[&::-webkit-slider-thumb]:h-4')
  })

  it('applies lg size classes', () => {
    const { container } = render(<Slider size="lg" />)
    const input = container.querySelector<HTMLInputElement>('input[type="range"]')!
    expect(input.classList.contains('h-3')).toBe(true)
    expect(input.className).toContain('[&::-webkit-slider-thumb]:h-5')
  })

  it('merges custom className', () => {
    const { container } = render(<Slider className="my-slider" />)
    expect(container.querySelector<HTMLInputElement>('input[type="range"]')!.classList.contains('my-slider')).toBe(true)
  })
})

describe('RangeSlider', () => {
  it('renders two range inputs', () => {
    const { container } = render(<RangeSlider />)
    expect(container.querySelectorAll<HTMLInputElement>('input[type="range"]')).toHaveLength(2)
  })

  it('renders a label and the value range', () => {
    render(<RangeSlider label="Range" />)
    expect(screen.getByText('Range')).toBeTruthy()
    expect(screen.getByText('0 – 100')).toBeTruthy()
  })

  it('calls onValueChange when the low input changes', () => {
    const onValueChange = vi.fn()
    const { container } = render(<RangeSlider onValueChange={onValueChange} />)
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="range"]')
    fireEvent.change(inputs[0], { target: { value: '30' } })
    expect(onValueChange).toHaveBeenCalledWith([30, 100])
  })

  it('calls onValueChange when the high input changes', () => {
    const onValueChange = vi.fn()
    const { container } = render(<RangeSlider onValueChange={onValueChange} />)
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="range"]')
    fireEvent.change(inputs[1], { target: { value: '80' } })
    expect(onValueChange).toHaveBeenCalledWith([0, 80])
  })

  it('updates the internal range when uncontrolled', () => {
    const { container } = render(<RangeSlider label="Range" defaultValue={[10, 90]} />)
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="range"]')
    fireEvent.change(inputs[0], { target: { value: '40' } })
    expect(screen.getByText('40 – 90')).toBeTruthy()
  })

  it('respects a controlled value', () => {
    render(<RangeSlider label="Range" value={[10, 90]} />)
    expect(screen.getByText('10 – 90')).toBeTruthy()
  })

  it('formats range values with formatValue', () => {
    render(<RangeSlider label="Range" formatValue={(v) => `${v}%`} />)
    expect(screen.getByText('0% – 100%')).toBeTruthy()
  })

  it('shows the value range without a label when showValue is true', () => {
    render(<RangeSlider value={[20, 80]} showValue />)
    expect(screen.getByText('20 – 80')).toBeTruthy()
  })

  it('hides the value range without a label when showValue is false', () => {
    render(<RangeSlider value={[20, 80]} />)
    expect(screen.queryByText('20 – 80')).toBeNull()
  })

  it('renders the filled range styles', () => {
    const { container } = render(<RangeSlider value={[20, 80]} />)
    const fill = container.querySelector<HTMLElement>('div.bg-primary')!
    expect(fill.style.left).toBe('20%')
    expect(fill.style.right).toBe('20%')
  })
})
