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

import { AutoTrainConfigCard } from './AutoTrainConfigCard'

afterEach(() => cleanup())

describe('AutoTrainConfigCard', () => {
  it('renders config form', () => {
    render(<AutoTrainConfigCard onSave={vi.fn()} />)
    expect(screen.getByText('Configuration')).toBeTruthy()
    expect(screen.getByText('Pair Threshold')).toBeTruthy()
    expect(screen.getByText('Check Interval (seconds)')).toBeTruthy()
    expect(screen.getByText('Save Configuration')).toBeTruthy()
  })

  it('shows default values', () => {
    render(<AutoTrainConfigCard onSave={vi.fn()} />)
    expect(screen.getByTestId('threshold-input')).toHaveValue(10)
    expect(screen.getByTestId('interval-input')).toHaveValue(120)
  })

  it('shows custom values', () => {
    render(<AutoTrainConfigCard threshold={50} intervalS={300} onSave={vi.fn()} />)
    expect(screen.getByTestId('threshold-input')).toHaveValue(50)
    expect(screen.getByTestId('interval-input')).toHaveValue(300)
  })

  it('updates threshold', () => {
    render(<AutoTrainConfigCard />)
    fireEvent.change(screen.getByTestId('threshold-input'), { target: { value: '25' } })
    expect(screen.getByTestId('threshold-input')).toHaveValue(25)
  })

  it('calls onSave', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    render(<AutoTrainConfigCard onSave={onSave} />)
    fireEvent.change(screen.getByTestId('threshold-input'), { target: { value: '30' } })
    fireEvent.click(screen.getByText('Save Configuration'))
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(30, 120)
    })
  })

  it('shows saving state', async () => {
    const onSave = vi.fn().mockImplementation(() => new Promise(r => setTimeout(r, 100)))
    render(<AutoTrainConfigCard onSave={onSave} />)
    fireEvent.click(screen.getByText('Save Configuration'))
    expect(screen.getByText('Saving...')).toBeTruthy()
  })
})
