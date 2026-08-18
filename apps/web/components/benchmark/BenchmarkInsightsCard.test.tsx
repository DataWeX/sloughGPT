// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { BenchmarkInsightsCard } from './BenchmarkInsightsCard'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

afterEach(() => { cleanup() })

const quality = {
  coherence_score: 0.85,
  quality_score: 0.72,
  repetition_rate: 0.15,
}

const metrics: BenchmarkResult = {
  model: 'gpt2',
  perplexity: 12.5,
  throughput_tokens_per_sec: 8.3,
  latency_p50_ms: 3200,
  memory_mb: 500,
  num_parameters: 124_000_000,
  latency_ms: 3200,
  throughput: 8.3,
  inference_time_ms: 3000,
}

const stats = { total: 42, avg_tokens: 64 }

describe('BenchmarkInsightsCard', () => {
  it('returns null for all null props', () => {
    const { container } = render(<BenchmarkInsightsCard metrics={null} quality={null} stats={null} />)
    expect(container.querySelector('[data-testid="benchmark-insights"]')).toBeNull()
  })

  it('renders card with quality data', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByTestId('benchmark-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Performance Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows quality score', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByText('72%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows coherence score', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByText('85%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows quality label', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByText('Good').length).toBeGreaterThanOrEqual(1)
  })

  it('shows coherence label', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByText('Excellent').length).toBeGreaterThanOrEqual(1)
  })

  it('shows perplexity', () => {
    render(<BenchmarkInsightsCard metrics={metrics} quality={null} stats={null} />)
    expect(screen.getAllByText('12.50').length).toBeGreaterThanOrEqual(1)
  })

  it('shows throughput', () => {
    render(<BenchmarkInsightsCard metrics={metrics} quality={null} stats={null} />)
    expect(screen.getAllByText('8.3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('tokens/sec').length).toBeGreaterThanOrEqual(1)
  })

  it('shows low repetition message', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={quality} stats={null} />)
    expect(screen.getAllByText(/Low repetition rate/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows high repetition warning', () => {
    const highRep = { ...quality, repetition_rate: 0.5 }
    render(<BenchmarkInsightsCard metrics={null} quality={highRep} stats={null} />)
    expect(screen.getAllByText(/High repetition rate/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows stats', () => {
    render(<BenchmarkInsightsCard metrics={null} quality={null} stats={stats} />)
    expect(screen.getAllByText(/42 responses logged/).length).toBeGreaterThanOrEqual(1)
  })
})
