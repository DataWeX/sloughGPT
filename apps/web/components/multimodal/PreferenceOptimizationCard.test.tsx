import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

import PreferenceOptimizationCard from './PreferenceOptimizationCard'

const DPOCard = PreferenceOptimizationCard

describe('PreferenceOptimizationCard', () => {
  afterEach(cleanup)

  const base = {
    dpoRunning: false, dpoStatus: '', dpoResult: null, dpoError: null,
    dpoAccepted: 0, dpoRejected: 0, onTrigger: vi.fn(),
  }

  it('renders DPO description and trigger button', () => {
    render(<PreferenceOptimizationCard {...base} />)
    expect(screen.getByText(/Direct Preference Optimization/)).toBeDefined()
    expect(screen.getByText('Run DPO')).toBeDefined()
  })

  it('calls onTrigger on button click', () => {
    render(<PreferenceOptimizationCard {...base} />)
    fireEvent.click(screen.getByText('Run DPO'))
    expect(base.onTrigger).toHaveBeenCalledOnce()
  })

  it('shows running state when dpoRunning is true', () => {
    render(<PreferenceOptimizationCard {...base} dpoRunning />)
    expect(screen.getByText('Running…')).toBeDefined()
  })

  it('shows accepted result', () => {
    const result = { status: 'accepted', steps: 50, avg_loss: 0.42, ppl_before: 20, ppl_after: 15, ppl_delta_pct: -25, pairs_trained: 30, elapsed_seconds: 120 }
    render(<PreferenceOptimizationCard {...base} dpoResult={result} />)
    expect(screen.getByText(/DPO accepted/)).toBeDefined()
    expect(screen.getByText(/50 steps/)).toBeDefined()
    expect(screen.getByText(/avg loss 0.4200/)).toBeDefined()
  })

  it('shows rejected result', () => {
    const result = { status: 'rejected' }
    render(<PreferenceOptimizationCard {...base} dpoResult={result} />)
    expect(screen.getByText(/DPO rejected/)).toBeDefined()
  })

  it('shows error message', () => {
    render(<PreferenceOptimizationCard {...base} dpoError="Something went wrong" />)
    expect(screen.getByText('Something went wrong')).toBeDefined()
  })

  it('shows accepted/rejected counts', () => {
    render(<PreferenceOptimizationCard {...base} dpoAccepted={10} dpoRejected={3} />)
    expect(screen.getByText(/10 accepted/)).toBeDefined()
    expect(screen.getByText(/3 rejected/)).toBeDefined()
  })

  it('disables button when running', () => {
    render(<PreferenceOptimizationCard {...base} dpoRunning />)
    expect(screen.getByText('Running…')).toBeDefined()
    expect(screen.getByText('Running…')).toBeDisabled()
  })
})
