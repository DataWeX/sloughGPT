import { describe, it, expect, vi, afterEach } from 'vitest'
import React, { createContext, useContext } from 'react'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const SelectCtx = createContext<(v: string) => void>(() => {})

vi.mock('@sloughgpt/strui', () => {
  const MockSelect = ({ value, onValueChange, children }: any) => (
    <SelectCtx.Provider value={onValueChange}>
      <div data-testid="select" data-value={value}>{children}</div>
    </SelectCtx.Provider>
  )
  const MockSelectTrigger = ({ children, className, 'aria-label': ariaLabel }: any) => (
    <button className={className} aria-label={ariaLabel} type="button">{children}</button>
  )
  const MockSelectValue = () => <span data-testid="select-value" />
  const MockSelectContent = ({ children }: any) => <div>{children}</div>
  const MockSelectItem = ({ value, children }: any) => {
    const onValueChange = useContext(SelectCtx)
    return (
      <button type="button" data-value={value} onClick={() => onValueChange(value)}>
        {children}
      </button>
    )
  }
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: ({ children }: any) => <div>{children}</div>,
    CardContent: ({ children }: any) => <div>{children}</div>,
    CardHeader: ({ children }: any) => <div>{children}</div>,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Input: ({ value, onChange, className, placeholder, min, step, type }: any) => (
      <input value={value} onChange={onChange} className={className} placeholder={placeholder} min={min} step={step} type={type} />
    ),
    Select: MockSelect,
    SelectTrigger: MockSelectTrigger,
    SelectValue: MockSelectValue,
    SelectContent: MockSelectContent,
    SelectItem: MockSelectItem,
  }
})

import { CheckpointFilterBar } from './CheckpointFilterBar'

const noop = () => {}

function makeProps(overrides: any = {}) {
  return {
    types: ['auto', 'manual'],
    typeFilter: 'all',
    onTypeFilterChange: noop,
    lossMax: '',
    onLossMaxChange: noop,
    total: 5,
    shown: 3,
    ...overrides,
  }
}

afterEach(() => cleanup())

describe('CheckpointFilterBar', () => {
  it('renders nothing when fewer than 3 checkpoints exist', () => {
    const { container } = render(<CheckpointFilterBar {...makeProps({ total: 2 })} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders the filter card and counts', () => {
    render(<CheckpointFilterBar {...makeProps()} />)
    expect(screen.getByText('Filter checkpoints')).toBeTruthy()
    expect(screen.getByText('3/5 shown')).toBeTruthy()
  })

  it('renders each type as a selectable option', () => {
    render(<CheckpointFilterBar {...makeProps()} />)
    const item = screen.getByRole('button', { name: 'auto' })
    fireEvent.click(item)
  })

  it('fires onTypeFilterChange when a type option is selected', () => {
    const onTypeFilterChange = vi.fn()
    render(<CheckpointFilterBar {...makeProps({ onTypeFilterChange })} />)
    fireEvent.click(screen.getByRole('button', { name: 'auto' }))
    expect(onTypeFilterChange).toHaveBeenCalledWith('auto')
  })

  it('fires onLossMaxChange when the loss input changes', () => {
    const onLossMaxChange = vi.fn()
    render(<CheckpointFilterBar {...makeProps({ onLossMaxChange })} />)
    fireEvent.change(screen.getByPlaceholderText('e.g. 2.0'), { target: { value: '2.5' } })
    expect(onLossMaxChange).toHaveBeenCalledWith('2.5')
  })

  it('hides the clear button when no filters are active', () => {
    render(<CheckpointFilterBar {...makeProps()} />)
    expect(screen.queryByText('Clear filters')).toBeNull()
  })

  it('shows the clear button when a type filter is active', () => {
    render(<CheckpointFilterBar {...makeProps({ typeFilter: 'auto' })} />)
    expect(screen.getByText('Clear filters')).toBeTruthy()
  })

  it('shows the clear button when a loss filter is active', () => {
    render(<CheckpointFilterBar {...makeProps({ lossMax: '2.0' })} />)
    expect(screen.getByText('Clear filters')).toBeTruthy()
  })

  it('resets both filters when clear is clicked', () => {
    const onTypeFilterChange = vi.fn()
    const onLossMaxChange = vi.fn()
    render(<CheckpointFilterBar {...makeProps({ typeFilter: 'auto', lossMax: '2.0', onTypeFilterChange, onLossMaxChange })} />)
    fireEvent.click(screen.getByText('Clear filters'))
    expect(onTypeFilterChange).toHaveBeenCalledWith('all')
    expect(onLossMaxChange).toHaveBeenCalledWith('')
  })

  it('selects the current type value in the select control', () => {
    render(<CheckpointFilterBar {...makeProps({ typeFilter: 'manual' })} />)
    expect(screen.getByTestId('select').getAttribute('data-value')).toBe('manual')
  })
})
