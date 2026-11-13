import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ModelUsageCard from './ModelUsageCard'

describe('ModelUsageCard', () => {
  afterEach(cleanup)

  it('renders title and KPI labels when online', () => {
    render(<ModelUsageCard inferenceCount={10} requestCount={20} modelType="gpt2" isOnline />)
    expect(screen.getByText('Usage Statistics')).toBeDefined()
    expect(screen.getByText('Inferences')).toBeDefined()
    expect(screen.getByText('Requests')).toBeDefined()
    expect(screen.getByText('Active Model')).toBeDefined()
    expect(screen.getByText('Avg Tokens')).toBeDefined()
  })

  it('hides the card when offline', () => {
    render(<ModelUsageCard inferenceCount={0} requestCount={0} modelType={null} isOnline={false} />)
    expect(screen.queryByText('Usage Statistics')).toBeNull()
    expect(screen.queryByText('Inferences')).toBeNull()
  })

  it('shows formatted inference and request counts', () => {
    render(<ModelUsageCard inferenceCount={1234} requestCount={56789} modelType="gpt2" isOnline />)
    expect(screen.getByText('1,234')).toBeDefined()
    expect(screen.getByText('56,789')).toBeDefined()
  })

  it('shows the active model type', () => {
    render(<ModelUsageCard inferenceCount={1} requestCount={1} modelType="qwen-0.5b" isOnline />)
    expect(screen.getByText('qwen-0.5b')).toBeDefined()
  })

  it('shows None when no model type', () => {
    render(<ModelUsageCard inferenceCount={1} requestCount={1} modelType={null} isOnline />)
    expect(screen.getByText('None')).toBeDefined()
  })

  it('computes average tokens as the rounded request-to-inference ratio', () => {
    render(<ModelUsageCard inferenceCount={10} requestCount={25} modelType="gpt2" isOnline />)
    expect(screen.getByText('3')).toBeDefined()
  })

  it('shows a dash for average tokens when inference count is zero', () => {
    render(<ModelUsageCard inferenceCount={0} requestCount={25} modelType="gpt2" isOnline />)
    expect(screen.getByText('—')).toBeDefined()
  })
})
