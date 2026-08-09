import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import ModelUsageCard from './ModelUsageCard'

afterEach(() => { cleanup() })

describe('ModelUsageCard', () => {
  it('returns null when offline', () => {
    const { container } = render(
      <ModelUsageCard inferenceCount={0} requestCount={0} modelType={null} isOnline={false} />,
    )
    expect(container.querySelector('[class*="rounded-lg"]')).toBeNull()
    expect(screen.queryAllByText('Usage Statistics').length).toBe(0)
  })

  it('renders a card when online', () => {
    const { container } = render(
      <ModelUsageCard inferenceCount={1} requestCount={1} modelType={null} isOnline={true} />,
    )
    expect(container.querySelector('[class*="rounded-lg"]')).not.toBeNull()
  })

  it('renders title and KPI labels when online', () => {
    render(
      <ModelUsageCard inferenceCount={10} requestCount={20} modelType="gpt2" isOnline={true} />,
    )
    expect(screen.getAllByText('Usage Statistics').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Inferences').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Requests').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Active Model').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Avg Tokens').length).toBeGreaterThanOrEqual(1)
  })

  it('shows formatted inference and request counts', () => {
    render(
      <ModelUsageCard inferenceCount={1234} requestCount={56789} modelType="gpt2" isOnline={true} />,
    )
    expect(screen.getAllByText('1,234').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('56,789').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model type when provided', () => {
    render(
      <ModelUsageCard inferenceCount={1} requestCount={1} modelType="qwen-0.5b" isOnline={true} />,
    )
    expect(screen.getAllByText('qwen-0.5b').length).toBeGreaterThanOrEqual(1)
  })

  it('shows None when model type is null', () => {
    render(
      <ModelUsageCard inferenceCount={1} requestCount={1} modelType={null} isOnline={true} />,
    )
    expect(screen.getAllByText('None').length).toBeGreaterThanOrEqual(1)
  })

  it('computes average tokens as rounded request/inference ratio', () => {
    render(
      <ModelUsageCard inferenceCount={10} requestCount={25} modelType="gpt2" isOnline={true} />,
    )
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows dash for average tokens when inference count is zero', () => {
    render(
      <ModelUsageCard inferenceCount={0} requestCount={25} modelType="gpt2" isOnline={true} />,
    )
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(1)
  })
})
