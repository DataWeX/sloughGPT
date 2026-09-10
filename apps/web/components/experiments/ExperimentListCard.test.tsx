// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { ExperimentListCard } from './ExperimentListCard'

afterEach(() => cleanup())

const mockExps = [
  { id: 'exp-1', name: 'LR Test', runs: 5, status: 'running' },
  { id: 'exp-2', name: 'Batch Size', runs: 3, status: 'completed' },
  { id: 'exp-3', name: 'Dropout Sweep', status: 'queued' },
]

describe('ExperimentListCard', () => {
  it('renders empty state', () => {
    render(<ExperimentListCard experiments={[]} />)
    expect(screen.getByText('Experiments')).toBeTruthy()
    expect(screen.getByText('(0)')).toBeTruthy()
    expect(screen.getByText('No experiments yet.')).toBeTruthy()
  })

  it('shows experiment list', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('LR Test')).toBeTruthy()
    expect(screen.getByText('Batch Size')).toBeTruthy()
  })

  it('shows status badges', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('queued')).toBeTruthy()
  })

  it('shows run counts', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('5 runs')).toBeTruthy()
    expect(screen.getByText('3 runs')).toBeTruthy()
  })

  it('filters by search', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    fireEvent.change(screen.getByTestId('experiment-search'), { target: { value: 'LR' } })
    expect(screen.getByText('LR Test')).toBeTruthy()
    expect(screen.queryByText('Batch Size')).toBeNull()
  })

  it('calls onSelect', () => {
    const onSelect = vi.fn()
    render(<ExperimentListCard experiments={mockExps} onSelect={onSelect} />)
    const exp1 = screen.getByTestId('experiment-exp-1')
    const btn = exp1.querySelector('button')
    if (btn) fireEvent.click(btn)
    expect(onSelect).toHaveBeenCalledWith('exp-1')
  })

  it('calls onDelete', () => {
    const onDelete = vi.fn()
    render(<ExperimentListCard experiments={mockExps} onDelete={onDelete} />)
    const delBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Del')
    fireEvent.click(delBtns[0])
    expect(onDelete).toHaveBeenCalledWith('exp-1')
  })

  it('highlights selected experiment', () => {
    render(<ExperimentListCard experiments={mockExps} selectedId="exp-2" />)
    const exp2 = screen.getByTestId('experiment-exp-2')
    expect(exp2.className).toContain('border-primary/50')
  })
})
