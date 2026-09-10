// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
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

import { AutoTrainStatusCard } from './AutoTrainStatusCard'

afterEach(() => cleanup())

describe('AutoTrainStatusCard', () => {
  it('renders with null data', () => {
    render(<AutoTrainStatusCard status={null} stats={null} />)
    expect(screen.getByText('Status')).toBeTruthy()
    expect(screen.getByText('Disabled')).toBeTruthy()
    expect(screen.getByText('10')).toBeTruthy()
  })

  it('shows enabled status', () => {
    render(<AutoTrainStatusCard
      status={{ enabled: true, threshold: 20, pending_count: 5, last_train: null }}
      stats={{ total: 100, pending: 5, synced: 90, used: 80, by_quality: { good: 70, bad: 10 } }}
    />)
    expect(screen.getByText('Enabled')).toBeTruthy()
    expect(screen.getByText('20')).toBeTruthy()
    expect(screen.getByText('5')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
  })

  it('shows quality breakdown', () => {
    render(<AutoTrainStatusCard
      status={{ enabled: true, threshold: 10, pending_count: 0, last_train: null }}
      stats={{ total: 50, pending: 0, synced: 50, used: 50, by_quality: { good: 40, bad: 15 } }}
    />)
    expect(screen.getByText('Quality Breakdown')).toBeTruthy()
    expect(screen.getByText('good')).toBeTruthy()
    expect(screen.getByText('bad')).toBeTruthy()
    expect(screen.getByText('40')).toBeTruthy()
    expect(screen.getByText('15')).toBeTruthy()
  })
})
