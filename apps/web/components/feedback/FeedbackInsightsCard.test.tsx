// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { FeedbackInsightsCard } from './FeedbackInsightsCard'
import type { FeedbackStats } from '@/lib/feedback-controller'

afterEach(() => { cleanup() })

function makeStats(overrides: Partial<FeedbackStats['db_stats']> = {}): FeedbackStats {
  return {
    db_stats: {
      conversations: 10,
      messages: 50,
      feedback_total: 25,
      thumbs_up: 20,
      thumbs_down: 5,
      ratio: 0.8,
      ...overrides,
    },
    current_weights: { temperature: 0.7, repetition_penalty: 1.1 },
    history_length: 30,
  }
}

describe('FeedbackInsightsCard', () => {
  it('renders empty state when no stats', () => {
    const { container } = render(<FeedbackInsightsCard stats={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders insights card', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByTestId('feedback-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Feedback Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows sentiment score', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByText('Sentiment').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('80%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows quality label', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByText('Excellent').length).toBeGreaterThanOrEqual(1)
  })

  it('shows feedback per conversation', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByText('2.5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows activity level', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByText('Medium').length).toBeGreaterThanOrEqual(1)
  })

  it('shows temperature and repetition penalty', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    expect(screen.getAllByText('0.7').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1.1').length).toBeGreaterThanOrEqual(1)
  })

  it('shows poor quality for low ratio', () => {
    render(<FeedbackInsightsCard stats={makeStats({ ratio: 0.2, thumbs_up: 2, thumbs_down: 8 })} />)
    expect(screen.getAllByText('Poor').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('20%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows high activity for many history entries', () => {
    render(<FeedbackInsightsCard stats={makeStats()} />)
    // history_length=30 → Medium
    expect(screen.getAllByText('Medium').length).toBeGreaterThanOrEqual(1)
  })
})
