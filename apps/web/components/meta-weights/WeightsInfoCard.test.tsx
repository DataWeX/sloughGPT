// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { WeightsInfoCard } from './WeightsInfoCard'

afterEach(() => { cleanup() })

describe('WeightsInfoCard', () => {
  it('renders title', () => {
    render(<WeightsInfoCard />)
    expect(screen.getByText('How It Works')).toBeDefined()
  })

  it('renders explanation text', () => {
    render(<WeightsInfoCard />)
    expect(screen.getByText(/Meta-weights adjust inference parameters/)).toBeDefined()
  })

  it('renders temperature description', () => {
    render(<WeightsInfoCard />)
    expect(screen.getByText(/Temperature/)).toBeDefined()
    expect(screen.getByText(/controls randomness/)).toBeDefined()
  })

  it('renders all parameter items', () => {
    render(<WeightsInfoCard />)
    expect(screen.getByText(/Top P/)).toBeDefined()
    expect(screen.getByText(/Repetition Penalty/)).toBeDefined()
    expect(screen.getByText(/Style Bias/)).toBeDefined()
    expect(screen.getByText(/Confidence Boost/)).toBeDefined()
  })

  it('renders list of parameters', () => {
    const { container } = render(<WeightsInfoCard />)
    const listItems = container.querySelectorAll('li')
    expect(listItems.length).toBe(5)
  })
})
