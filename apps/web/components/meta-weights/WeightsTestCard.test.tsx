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
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { WeightsTestCard } from './WeightsTestCard'

afterEach(() => { cleanup() })

describe('WeightsTestCard', () => {
  it('renders title', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Test Weights')).toBeDefined()
  })

  it('renders message input', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByPlaceholderText('Type a message to test...')).toBeDefined()
  })

  it('renders Compute button', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Compute')).toBeDefined()
  })

  it('disables Compute when message is empty', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Compute').hasAttribute('disabled')).toBe(true)
  })

  it('shows computing state', () => {
    render(<WeightsTestCard testMessage="hello" onMessageChange={() => {}} onCompute={() => {}} testing />)
    expect(screen.getByText('Computing...')).toBeDefined()
  })

  it('renders weight bars when weights provided', () => {
    render(
      <WeightsTestCard
        testMessage="test"
        onMessageChange={() => {}}
        onCompute={() => {}}
        weights={{
          temperature: 0.8,
          top_p: 0.9,
          repetition_penalty: 1.2,
          style_bias: 0.5,
          confidence_boost: 0.7,
          top_k: 40,
          based_on_samples: 15,
        }}
      />
    )
    expect(screen.getByText('Temperature')).toBeDefined()
    expect(screen.getByText('Top P')).toBeDefined()
    expect(screen.getByText('Repetition Penalty')).toBeDefined()
    expect(screen.getByText('Style Bias')).toBeDefined()
    expect(screen.getByText('Confidence Boost')).toBeDefined()
    expect(screen.getByText('Top K')).toBeDefined()
  })

  it('renders sample count', () => {
    render(
      <WeightsTestCard
        testMessage="test"
        onMessageChange={() => {}}
        onCompute={() => {}}
        weights={{
          temperature: 0.8,
          top_p: 0.9,
          repetition_penalty: 1.2,
          style_bias: 0.5,
          confidence_boost: 0.7,
          top_k: 40,
          based_on_samples: 15,
        }}
      />
    )
    expect(screen.getByText('Based on 15 feedback samples')).toBeDefined()
  })
})
