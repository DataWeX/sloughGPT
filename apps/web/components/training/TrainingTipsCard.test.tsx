// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

import { TrainingTipsCard } from './TrainingTipsCard'
import type { Checkpoint } from '@/lib/souls-controller'

afterEach(() => { cleanup() })

function makeCheckpoint(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'cp1', soul: 'test', ...overrides }
}

describe('TrainingTipsCard', () => {
  it('renders without crashing with empty checkpoints', () => {
    render(<TrainingTipsCard checkpoints={[]} />)
    expect(screen.getByText('Training tips')).toBeDefined()
    expect(screen.getByText('Start training to see personalized tips and recommendations.')).toBeDefined()
  })

  it('shows loading skeleton when loading and no checkpoints', () => {
    const { container } = render(<TrainingTipsCard checkpoints={[]} loading />)
    expect(screen.getByText('Training tips')).toBeDefined()
    const pulse = container.querySelector('.animate-pulse')
    expect(pulse).toBeTruthy()
  })

  it('shows info tip for small number of checkpoints', () => {
    const checkpoints = [makeCheckpoint({ loss: 1.0 })]
    render(<TrainingTipsCard checkpoints={checkpoints} />)
    expect(screen.getByText(/Small number of checkpoints/)).toBeDefined()
  })

  it('shows warning when loss gap is large', () => {
    const checkpoints = [
      makeCheckpoint({ loss: 2.0 }),
      makeCheckpoint({ loss: 1.0 }),
      makeCheckpoint({ loss: 0.5 }),
    ]
    render(<TrainingTipsCard checkpoints={checkpoints} />)
    expect(screen.getByText(/Large gap between current/)).toBeDefined()
  })

  it('shows success when training has converged', () => {
    const checkpoints = Array.from({ length: 6 }, (_, i) =>
      makeCheckpoint({ loss: 0.5 + i * 0.001 })
    )
    render(<TrainingTipsCard checkpoints={checkpoints} />)
    expect(screen.getByText(/Training has converged/)).toBeDefined()
  })

  it('shows warning for low average quality', () => {
    const checkpoints = [
      makeCheckpoint({ loss: 1.0, avg_quality: 2.0 }),
      makeCheckpoint({ loss: 0.9, avg_quality: 2.0 }),
    ]
    render(<TrainingTipsCard checkpoints={checkpoints} />)
    expect(screen.getByText(/Low average quality/)).toBeDefined()
  })
})
