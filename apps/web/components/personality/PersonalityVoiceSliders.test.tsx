/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityVoiceSliders } from './PersonalityVoiceSliders'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const mockVoice = {
  formality: 0.5,
  warmth: 0.8,
  confidence: 0.7,
  humor: 0.3,
  verbosity: 0.6,
  empathy: 0.9,
}

describe('PersonalityVoiceSliders', () => {
  it('renders title "Voice"', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('Voice')).toBeDefined()
  })

  it('renders six voice labels', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('Formality')).toBeDefined()
    expect(screen.getByText('Warmth')).toBeDefined()
    expect(screen.getByText('Confidence')).toBeDefined()
    expect(screen.getByText('Humor')).toBeDefined()
    expect(screen.getByText('Verbosity')).toBeDefined()
    expect(screen.getByText('Empathy')).toBeDefined()
  })

  it('renders six range inputs', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    const sliders = screen.getAllByRole('slider')
    expect(sliders).toHaveLength(6)
  })

  it('displays percentage values', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('50%')).toBeDefined()
    expect(screen.getByText('80%')).toBeDefined()
    expect(screen.getByText('90%')).toBeDefined()
  })

  it('renders card description', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('How the system sounds when communicating')).toBeDefined()
  })
})
