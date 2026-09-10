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
}))

import { RateLimitConfigCard } from './RateLimitConfigCard'

afterEach(() => cleanup())

describe('RateLimitConfigCard', () => {
  it('renders the card title', () => {
    render(<RateLimitConfigCard />)
    expect(screen.getAllByText('Configuration').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading text when no config', () => {
    render(<RateLimitConfigCard />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('displays config as JSON', () => {
    const config = { enabled: true, requests_per_minute: 60 }
    render(<RateLimitConfigCard config={config} />)
    expect(screen.getByText(/enabled/)).toBeTruthy()
    expect(screen.getByText(/requests_per_minute/)).toBeTruthy()
  })

  it('renders a pre element', () => {
    render(<RateLimitConfigCard config={{ a: 1 }} />)
    const pre = screen.getByText(/"a": 1/)
    expect(pre.tagName).toBe('PRE')
  })

  it('handles empty config object', () => {
    render(<RateLimitConfigCard config={{}} />)
    expect(screen.getByText('{}')).toBeTruthy()
  })
})
