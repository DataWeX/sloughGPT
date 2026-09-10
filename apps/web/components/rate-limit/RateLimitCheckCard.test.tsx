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
}))

import { RateLimitCheckCard } from './RateLimitCheckCard'
import type { RateLimitCheck } from './RateLimitCheckCard'

afterEach(() => cleanup())

const allowedResult: RateLimitCheck = { allowed: true, wait_time: 0 }
const deniedResult: RateLimitCheck = { allowed: false, wait_time: 12.3 }

describe('RateLimitCheckCard', () => {
  it('renders the card title', () => {
    render(<RateLimitCheckCard />)
    expect(screen.getAllByText('Check Rate Limit').length).toBeGreaterThanOrEqual(1)
  })

  it('renders the check button', () => {
    render(<RateLimitCheckCard />)
    expect(screen.getAllByText('Check Now').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checking state', () => {
    render(<RateLimitCheckCard checking />)
    expect(screen.getAllByText('Checking...').length).toBeGreaterThanOrEqual(1)
  })

  it('shows allowed result', () => {
    render(<RateLimitCheckCard checkResult={allowedResult} />)
    expect(screen.getByText('Allowed: Yes')).toBeTruthy()
  })

  it('shows denied result with wait time', () => {
    render(<RateLimitCheckCard checkResult={deniedResult} />)
    expect(screen.getByText('Allowed: No')).toBeTruthy()
    expect(screen.getByText('Wait: 12.3s')).toBeTruthy()
  })

  it('does not show wait time when allowed', () => {
    render(<RateLimitCheckCard checkResult={allowedResult} />)
    expect(screen.queryByText(/Wait:/)).toBeNull()
  })
})
