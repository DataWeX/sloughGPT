// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { EvalQualityCard } from './EvalQualityCard'

afterEach(() => cleanup())

describe('EvalQualityCard', () => {
  it('renders title', () => {
    render(<EvalQualityCard />)
    expect(screen.getByText('Quality Metrics')).toBeTruthy()
  })

  it('shows empty state when no data', () => {
    render(<EvalQualityCard />)
    expect(screen.getByText(/No quality data yet/)).toBeTruthy()
  })

  it('renders quality scores', () => {
    render(<EvalQualityCard
      coherenceScore={0.85}
      qualityScore={0.72}
      repetitionRate={0.15}
    />)
    expect(screen.getByText('85.0%')).toBeTruthy()
    expect(screen.getByText('72.0%')).toBeTruthy()
    expect(screen.getByText('15.0%')).toBeTruthy()
  })

  it('renders labels', () => {
    render(<EvalQualityCard
      coherenceScore={0.5}
      qualityScore={0.5}
      repetitionRate={0.5}
    />)
    expect(screen.getByText('Coherence')).toBeTruthy()
    expect(screen.getByText('Quality')).toBeTruthy()
    expect(screen.getByText('Repetition')).toBeTruthy()
  })
})
