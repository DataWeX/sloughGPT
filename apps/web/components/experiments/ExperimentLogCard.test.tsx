// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
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

import { ExperimentLogCard } from './ExperimentLogCard'

afterEach(() => cleanup())

describe('ExperimentLogCard', () => {
  it('renders nothing when no experiment selected', () => {
    const { container } = render(<ExperimentLogCard />)
    expect(container.firstChild).toBeNull()
  })

  it('renders log form when experiment selected', () => {
    render(<ExperimentLogCard experimentId="exp-1" onLogMetric={vi.fn()} onLogParam={vi.fn()} />)
    expect(screen.getByText('Log Data')).toBeTruthy()
    expect(screen.getByText('Metric')).toBeTruthy()
    expect(screen.getByText('Parameter')).toBeTruthy()
  })

  it('disables metric log when fields empty', () => {
    render(<ExperimentLogCard experimentId="exp-1" onLogMetric={vi.fn()} />)
    const logBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Log')
    expect(logBtns[0].hasAttribute('disabled')).toBe(true)
  })

  it('calls onLogMetric', async () => {
    const onLogMetric = vi.fn().mockResolvedValue(undefined)
    render(<ExperimentLogCard experimentId="exp-1" onLogMetric={onLogMetric} />)
    fireEvent.change(screen.getByTestId('metric-name'), { target: { value: 'accuracy' } })
    fireEvent.change(screen.getByTestId('metric-value'), { target: { value: '0.95' } })
    const logBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Log')
    fireEvent.click(logBtns[0])
    await waitFor(() => {
      expect(onLogMetric).toHaveBeenCalledWith('exp-1', 'accuracy', 0.95)
    })
  })

  it('calls onLogParam', async () => {
    const onLogParam = vi.fn().mockResolvedValue(undefined)
    render(<ExperimentLogCard experimentId="exp-1" onLogParam={onLogParam} />)
    fireEvent.change(screen.getByTestId('param-name'), { target: { value: 'lr' } })
    fireEvent.change(screen.getByTestId('param-value'), { target: { value: '0.001' } })
    const allBtns = screen.getAllByRole('button')
    const logBtn = allBtns.find(b => b.textContent === 'Log' && b.closest('[data-testid="experiment-log"]'))
    if (logBtn) fireEvent.click(logBtn)
    await waitFor(() => {
      expect(onLogParam).toHaveBeenCalledWith('exp-1', 'lr', '0.001')
    })
  })

  it('shows Complete button when onComplete provided', () => {
    render(<ExperimentLogCard experimentId="exp-1" onComplete={vi.fn()} />)
    expect(screen.getByText('Complete')).toBeTruthy()
  })

  it('calls onComplete', async () => {
    const onComplete = vi.fn().mockResolvedValue(undefined)
    render(<ExperimentLogCard experimentId="exp-1" onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Complete'))
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith('exp-1')
    })
  })
})
