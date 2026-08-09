// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { GenerationConfigCard, GENERATION_DEFAULTS } from './GenerationConfigCard'

afterEach(() => { cleanup() })

const defaults = { temperature: 0.7, top_p: 0.85, top_k: 40, max_new_tokens: 300, repetition_penalty: 1.15 }

describe('GenerationConfigCard', () => {
  it('renders card with title', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} />)
    expect(screen.getAllByTestId('generation-config').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Generation Config').length).toBeGreaterThanOrEqual(1)
  })

  it('shows all slider labels', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} />)
    expect(screen.getAllByText('Temperature').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Top-P').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Top-K').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Max Tokens').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Repetition Penalty').length).toBeGreaterThanOrEqual(1)
  })

  it('hides repetition penalty when showRepetitionPenalty=false', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} showRepetitionPenalty={false} />)
    expect(screen.queryByText('Repetition Penalty')).toBeNull()
  })

  it('shows current values as badges', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} />)
    expect(screen.getAllByText('temp=0.70').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('p=0.85').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('k=40').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('tokens=300').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Reset button when values differ from defaults', () => {
    render(<GenerationConfigCard values={{ ...defaults, temperature: 1.0 }} onChange={() => {}} onReset={() => {}} />)
    expect(screen.getAllByText('Reset').length).toBeGreaterThanOrEqual(1)
  })

  it('hides Reset button when values match defaults', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} onReset={() => {}} />)
    expect(screen.queryByText('Reset')).toBeNull()
  })

  it('calls onReset when Reset clicked', () => {
    const onReset = vi.fn()
    render(<GenerationConfigCard values={{ ...defaults, temperature: 1.0 }} onChange={() => {}} onReset={onReset} />)
    fireEvent.click(screen.getAllByText('Reset')[0])
    expect(onReset).toHaveBeenCalledOnce()
  })

  it('does not call onChange without onReset when values match', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} />)
    expect(screen.queryByText('Reset')).toBeNull()
  })

  it('exports GENERATION_DEFAULTS', () => {
    expect(GENERATION_DEFAULTS.temperature).toBe(0.7)
    expect(GENERATION_DEFAULTS.top_p).toBe(0.85)
    expect(GENERATION_DEFAULTS.top_k).toBe(40)
    expect(GENERATION_DEFAULTS.max_new_tokens).toBe(300)
    expect(GENERATION_DEFAULTS.repetition_penalty).toBe(1.15)
  })

  it('renders compact layout', () => {
    render(<GenerationConfigCard values={defaults} onChange={() => {}} compact />)
    expect(screen.getAllByTestId('generation-config').length).toBeGreaterThanOrEqual(1)
  })
})
