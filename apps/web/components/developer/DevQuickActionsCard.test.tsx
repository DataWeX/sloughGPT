// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { DevQuickActionsCard } from './DevQuickActionsCard'

afterEach(() => cleanup())

describe('DevQuickActionsCard', () => {
  it('renders all 6 actions', () => {
    render(<DevQuickActionsCard />)
    expect(screen.getByText('Restart Backend')).toBeTruthy()
    expect(screen.getByText('Clear Cache')).toBeTruthy()
    expect(screen.getByText('Reset Metrics')).toBeTruthy()
    expect(screen.getByText('Reload Models')).toBeTruthy()
    expect(screen.getByText('Health Check')).toBeTruthy()
    expect(screen.getByText('List Endpoints')).toBeTruthy()
  })

  it('shows descriptions for each action', () => {
    render(<DevQuickActionsCard />)
    expect(screen.getByText('Restart the FastAPI server')).toBeTruthy()
    expect(screen.getByText('Clear model cache')).toBeTruthy()
    expect(screen.getByText('Check system health')).toBeTruthy()
  })

  it('shows method and endpoint for each action', () => {
    render(<DevQuickActionsCard />)
    expect(screen.getByText('POST /system/restart')).toBeTruthy()
    expect(screen.getByText('GET /health')).toBeTruthy()
    expect(screen.getByText('GET /routes')).toBeTruthy()
  })
})
