import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ModelStatusCard from './ModelStatusCard'

describe('ModelStatusCard', () => {
  afterEach(cleanup)

  const base = {
    isOnline: true,
    health: { status: 'healthy', model_loaded: true, model_type: 'gpt2', inference_count: 42, summary: 'ok' },
    currentSoul: 'friendly',
    activeCheckpoint: null,
    modelsCount: 5,
    soulsCount: 3,
    checkpointsCount: 0,
    modelsLoading: false,
    soulsLoading: false,
    checkpointsLoading: false,
  }

  it('renders active pipeline when online', () => {
    render(<ModelStatusCard {...base} />)
    expect(screen.getByText('Active Pipeline')).toBeDefined()
    expect(screen.getByText('GPT 2')).toBeDefined()
    expect(screen.getByText('friendly')).toBeDefined()
  })

  it('shows inference count', () => {
    render(<ModelStatusCard {...base} />)
    expect(screen.getByText('42 inferences')).toBeDefined()
  })

  it('shows loading state in KPIs when loading', () => {
    render(<ModelStatusCard {...base} modelsLoading soulsLoading checkpointsLoading />)
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkpoints count', () => {
    render(<ModelStatusCard {...base} checkpointsCount={7} />)
    expect(screen.getByText('7')).toBeDefined()
  })

  it('renders KPIs with correct values', () => {
    render(<ModelStatusCard {...base} />)
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('3')).toBeDefined()
  })

  it('hides pipeline card when offline', () => {
    render(<ModelStatusCard {...base} isOnline={false} />)
    expect(screen.queryByText('Active Pipeline')).toBeNull()
  })
})
