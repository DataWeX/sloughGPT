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

import { SentimentTrendCard } from './SentimentTrendCard'

afterEach(() => { cleanup() })

describe('SentimentTrendCard', () => {
  it('renders empty state', () => {
    render(<SentimentTrendCard history={[]} />)
    expect(screen.getByText('Sentiment Trend')).toBeTruthy()
    expect(screen.getByText('No feedback history yet.')).toBeTruthy()
  })

  it('renders stats with positive feedback', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_down' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText('Sentiment Trend')).toBeTruthy()
    expect(screen.getByText(/66\.7% positive/)).toBeTruthy()
  })

  it('shows improving trend when recent is better', () => {
    const now = Date.now()
    const history = [
      { timestamp: new Date(now - 10 * 86400000).toISOString(), rating: 'thumbs_down' as const },
      { timestamp: new Date(now - 8 * 86400000).toISOString(), rating: 'thumbs_down' as const },
      { timestamp: new Date(now - 1 * 86400000).toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date(now - 1 * 86400000).toISOString(), rating: 'thumbs_up' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText(/improving/)).toBeTruthy()
  })

  it('shows thumbs up/down icons', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_down' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText('👍')).toBeTruthy()
    expect(screen.getByText('👎')).toBeTruthy()
  })

  it('shows quality score when present', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const, quality_score: 0.85 },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText(/Score: 85%/)).toBeTruthy()
  })
})
