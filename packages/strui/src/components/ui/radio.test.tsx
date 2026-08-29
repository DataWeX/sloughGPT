import { describe, expect, it, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Radio } from './radio'
import { RadioGroup } from './radio-group'

describe('Radio', () => {
  afterEach(() => cleanup())

  it('renders a radio input with role radio', () => {
    render(<Radio name="test" id="r1" />)
    const radios = screen.getAllByRole('radio')
    expect(radios).toHaveLength(1)
    expect(radios[0].getAttribute('type')).toBe('radio')
  })

  it('defaults to unchecked', () => {
    render(<Radio name="test" id="r2" />)
    expect((screen.getByRole('radio') as HTMLInputElement).checked).toBe(false)
  })

  it('reflects defaultChecked', () => {
    render(<Radio name="test" id="r3" defaultChecked />)
    expect((screen.getByRole('radio') as HTMLInputElement).checked).toBe(true)
  })

  it('fires onCheckedChange on click', () => {
    const onCheckedChange = vi.fn()
    render(<Radio name="test" id="r4" onCheckedChange={onCheckedChange} />)
    fireEvent.click(screen.getByRole('radio'))
    expect(onCheckedChange).toHaveBeenCalledWith(true)
  })

  it('is controlled by checked prop', () => {
    const onCheckedChange = vi.fn()
    render(<Radio name="test" id="r5" checked={true} onCheckedChange={onCheckedChange} />)
    expect((screen.getByRole('radio') as HTMLInputElement).checked).toBe(true)
  })

  it('renders disabled state', () => {
    render(<Radio name="test" id="r6" disabled />)
    expect((screen.getByRole('radio') as HTMLInputElement).disabled).toBe(true)
  })

  it('renders label and description', () => {
    render(<Radio name="test" id="r7" label="Option 1" description="First option" />)
    expect(screen.getByText('Option 1')).toBeTruthy()
    expect(screen.getByText('First option')).toBeTruthy()
    const label = screen.getByText('Option 1').closest('label')
    expect(label?.getAttribute('for')).toBe('r7')
  })

  it('applies size classes', () => {
    const { container } = render(<Radio name="test" id="r8" size="lg" />)
    const wrapper = container.querySelector('span')
    expect(wrapper?.className).toContain('h-5 w-5')
  })
})

describe('RadioGroup', () => {
  afterEach(() => cleanup())

  it('renders with radiogroup role', () => {
    render(
      <RadioGroup>
        <Radio value="a" id="rg1a" label="A" />
        <Radio value="b" id="rg1b" label="B" />
      </RadioGroup>,
    )
    expect(screen.getByRole('radiogroup')).toBeTruthy()
  })

  it('renders multiple radios', () => {
    render(
      <RadioGroup>
        <Radio value="a" id="rg2a" label="A" />
        <Radio value="b" id="rg2b" label="B" />
        <Radio value="c" id="rg2c" label="C" />
      </RadioGroup>,
    )
    expect(screen.getAllByRole('radio')).toHaveLength(3)
  })

  it('selects radio via onValueChange', () => {
    const onValueChange = vi.fn()
    render(
      <RadioGroup value="b" onValueChange={onValueChange}>
        <Radio value="a" id="rg3a" label="A" />
        <Radio value="b" id="rg3b" label="B" />
      </RadioGroup>,
    )
    const radios = screen.getAllByRole('radio')
    expect(radios[0].checked).toBe(false)
    expect(radios[1].checked).toBe(true)
    fireEvent.click(radios[0])
    expect(onValueChange).toHaveBeenCalledWith('a')
  })

  it('applies vertical layout by default', () => {
    const { container } = render(
      <RadioGroup>
        <Radio value="a" id="rg4a" />
        <Radio value="b" id="rg4b" />
      </RadioGroup>,
    )
    expect(container.firstChild?.className).toContain('flex-col')
  })

  it('applies horizontal layout', () => {
    const { container } = render(
      <RadioGroup orientation="horizontal">
        <Radio value="a" id="rg5a" />
        <Radio value="b" id="rg5b" />
      </RadioGroup>,
    )
    expect(container.firstChild?.className).toContain('flex-row')
  })
})
