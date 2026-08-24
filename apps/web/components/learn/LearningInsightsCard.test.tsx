// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { LearningInsightsCard } from './LearningInsightsCard'

afterEach(() => { cleanup() })

const facts = [
  { content: 'Python is a programming language', topic: 'programming', source: 'web', importance: 0.9 },
  { content: 'Machine learning uses data', topic: 'ml', source: 'web', importance: 0.7 },
  { content: 'Neural networks are inspired by brains', topic: 'ml', source: 'text', importance: 0.8 },
  { content: 'JavaScript runs in browsers', topic: 'programming', source: 'web', importance: 0.6 },
  { content: 'Deep learning is a subset of ML', topic: 'ml', source: 'text', importance: 0.85 },
]

describe('LearningInsightsCard', () => {
  it('renders empty state for empty facts', () => {
    render(<LearningInsightsCard facts={[]} />)
    expect(screen.getAllByText('No learning facts yet').length).toBeGreaterThanOrEqual(1)
  })

  it('renders insights card', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByTestId('learning-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Learning Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows fact count', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average importance', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByText('Avg Importance').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.77').length).toBeGreaterThanOrEqual(1)
  })

  it('shows high priority count', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByText('High Priority').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows topics breakdown', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByText('By Topic').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/ml/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/programming/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows sources breakdown', () => {
    render(<LearningInsightsCard facts={facts} />)
    expect(screen.getAllByText('By Source').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/web/).length).toBeGreaterThanOrEqual(1)
  })

  it('handles facts with empty topics', () => {
    render(<LearningInsightsCard facts={[
      { content: 'test', topic: '', source: 'test', importance: 0.5 },
    ]} />)
    expect(screen.getAllByText('untagged').length).toBeGreaterThanOrEqual(1)
  })

  it('computes average content length', () => {
    render(<LearningInsightsCard facts={[
      { content: 'ab', topic: 't', source: 's', importance: 0.5 },
      { content: 'abcd', topic: 't', source: 's', importance: 0.5 },
    ]} />)
    expect(screen.getAllByText('Avg Length').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3ch').length).toBeGreaterThanOrEqual(1)
  })
})
