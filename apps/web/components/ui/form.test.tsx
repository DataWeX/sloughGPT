/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Slider, RangeSlider, Toggle, FieldGroup, ToggleGroup, Tabs } from './form'

afterEach(cleanup)

describe('Slider', () => {
  it('renders range input', () => {
    render(<Slider value={50} onChange={() => {}} />)
    const el = screen.getByRole('slider') as HTMLInputElement
    expect(el.type).toBe('range')
  })

  it('displays current value', () => {
    const { container } = render(<Slider value={42} onChange={() => {}} label="Test" />)
    expect(screen.getByText('Test')).toBeInTheDocument()
    expect(container.textContent).toContain('42')
  })

  it('hides value when showValue is false', () => {
    render(<Slider value={50} onChange={() => {}} showValue={false} />)
    expect(screen.queryByText('50')).not.toBeInTheDocument()
  })

  it('formats value with formatValue', () => {
    render(<Slider value={0.5} onChange={() => {}} label="Test" formatValue={v => `${v * 100}%`} />)
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<Slider value={50} onChange={() => {}} label="Volume" />)
    expect(screen.getByText('Volume')).toBeInTheDocument()
  })

  it('calls onChange on input', () => {
    const onChange = vi.fn()
    render(<Slider value={50} onChange={onChange} min={0} max={100} />)
    fireEvent.change(screen.getByRole('slider'), { target: { value: '60' } })
    expect(onChange).toHaveBeenCalledWith(60)
  })

  it('passes min/max attributes', () => {
    render(<Slider value={5} onChange={() => {}} min={0} max={10} />)
    const el = screen.getByRole('slider') as HTMLInputElement
    expect(el.min).toBe('0')
    expect(el.max).toBe('10')
  })
})

describe('RangeSlider', () => {
  it('renders two range inputs', () => {
    render(<RangeSlider value={[20, 80]} onChange={() => {}} />)
    const sliders = screen.getAllByRole('slider')
    expect(sliders).toHaveLength(2)
  })

  it('displays formatted range', () => {
    render(<RangeSlider value={[20, 80]} onChange={() => {}} label="Range" />)
    expect(screen.getByText('20 - 80')).toBeInTheDocument()
  })

  it('displays formatted range with formatValue', () => {
    render(<RangeSlider value={[0.2, 0.8]} onChange={() => {}} label="Range" formatValue={v => `${(v * 100).toFixed(0)}%`} />)
    expect(screen.getByText('20% - 80%')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<RangeSlider value={[10, 90]} onChange={() => {}} label="Budget" />)
    expect(screen.getByText('Budget')).toBeInTheDocument()
  })

  it('calls onChange on low slider change', () => {
    const onChange = vi.fn()
    render(<RangeSlider value={[20, 80]} onChange={onChange} />)
    const [low] = screen.getAllByRole('slider')
    fireEvent.change(low, { target: { value: '30' } })
    expect(onChange).toHaveBeenCalled()
  })
})

describe('Toggle', () => {
  it('renders switch role', () => {
    render(<Toggle checked={false} onChange={() => {}} />)
    expect(screen.getByRole('switch')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<Toggle checked={false} onChange={() => {}} label="Dark mode" />)
    expect(screen.getByText('Dark mode')).toBeInTheDocument()
  })

  it('renders description when provided', () => {
    render(<Toggle checked={false} onChange={() => {}} description="Enable dark theme" />)
    expect(screen.getByText('Enable dark theme')).toBeInTheDocument()
  })

  it('sets aria-checked based on checked prop', () => {
    const { rerender } = render(<Toggle checked={true} onChange={() => {}} />)
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
    rerender(<Toggle checked={false} onChange={() => {}} />)
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'false')
  })

  it('calls onChange when clicked', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Toggle checked={false} onChange={onChange} />)
    await user.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('calls onChange with false when toggled off', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Toggle checked={true} onChange={onChange} />)
    await user.click(screen.getByRole('switch'))
    expect(onChange).toHaveBeenCalledWith(false)
  })

  it('is disabled when disabled prop is set', () => {
    render(<Toggle checked={false} onChange={() => {}} disabled />)
    expect(screen.getByRole('switch')).toBeDisabled()
  })
})

describe('FieldGroup', () => {
  it('renders children', () => {
    render(<FieldGroup><input data-testid="child" /></FieldGroup>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('renders label when provided', () => {
    render(<FieldGroup label="Email">content</FieldGroup>)
    expect(screen.getByText('Email')).toBeInTheDocument()
  })

  it('renders description when provided', () => {
    render(<FieldGroup description="Enter your email">content</FieldGroup>)
    expect(screen.getByText('Enter your email')).toBeInTheDocument()
  })

  it('renders error when provided', () => {
    render(<FieldGroup error="Invalid email">content</FieldGroup>)
    expect(screen.getByText('Invalid email')).toBeInTheDocument()
  })

  it('does not render error when absent', () => {
    const { container } = render(<FieldGroup>content</FieldGroup>)
    expect(container.querySelector('.text-destructive')).not.toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<FieldGroup className="my-4">content</FieldGroup>)
    expect(container.firstChild).toHaveClass('my-4')
  })
})

describe('ToggleGroup', () => {
  const options = [
    { value: 'day', label: 'Day' },
    { value: 'week', label: 'Week' },
    { value: 'month', label: 'Month' },
  ]

  it('renders all options', () => {
    render(<ToggleGroup value="day" onChange={() => {}} options={options} />)
    expect(screen.getByText('Day')).toBeInTheDocument()
    expect(screen.getByText('Week')).toBeInTheDocument()
    expect(screen.getByText('Month')).toBeInTheDocument()
  })

  it('highlights active option', () => {
    render(<ToggleGroup value="week" onChange={() => {}} options={options} />)
    const active = screen.getByText('Week').closest('button')
    expect(active).toHaveClass('bg-background')
    const inactive = screen.getByText('Day').closest('button')
    expect(inactive).toHaveClass('text-muted-foreground')
  })

  it('calls onChange on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ToggleGroup value="day" onChange={onChange} options={options} />)
    await user.click(screen.getByText('Week'))
    expect(onChange).toHaveBeenCalledWith('week')
  })

  it('renders icon when provided', () => {
    const opts = [{ value: 'x', label: 'X', icon: <span data-testid="icon" /> }]
    render(<ToggleGroup value="x" onChange={() => {}} options={opts} />)
    expect(screen.getByTestId('icon')).toBeInTheDocument()
  })
})

describe('Tabs (form)', () => {
  const tabs = [
    { value: 'a', label: 'Tab A' },
    { value: 'b', label: 'Tab B', count: 3 },
  ]

  it('renders all tab labels', () => {
    render(<Tabs value="a" onChange={() => {}} tabs={tabs} />)
    expect(screen.getByText('Tab A')).toBeInTheDocument()
    expect(screen.getByText('Tab B')).toBeInTheDocument()
  })

  it('highlights active tab', () => {
    render(<Tabs value="b" onChange={() => {}} tabs={tabs} />)
    expect(screen.getByText('Tab B').closest('button')).toHaveClass('bg-background')
    expect(screen.getByText('Tab A').closest('button')).toHaveClass('text-muted-foreground')
  })

  it('shows count when provided', () => {
    render(<Tabs value="a" onChange={() => {}} tabs={tabs} />)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('calls onChange on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Tabs value="a" onChange={onChange} tabs={tabs} />)
    await user.click(screen.getByText('Tab B'))
    expect(onChange).toHaveBeenCalledWith('b')
  })
})
