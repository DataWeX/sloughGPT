// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, ...p }: any) => <button onClick={onClick} {...p}>{children}</button>,
}))

import { OnboardingCard } from './OnboardingCard'

beforeEach(() => {
  vi.clearAllMocks()
  Storage.prototype.getItem = vi.fn(() => null)
  Storage.prototype.setItem = vi.fn()
})

afterEach(() => cleanup())

describe('OnboardingCard', () => {
  it('renders welcome step', () => {
    render(<OnboardingCard onComplete={vi.fn()} />)
    expect(screen.getByText('Welcome to SloughGPT')).toBeDefined()
  })

  it('calls onComplete immediately if already onboarded', () => {
    Storage.prototype.getItem = vi.fn(() => 'true')
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    expect(onComplete).toHaveBeenCalled()
  })

  it('advances through all steps', () => {
    render(<OnboardingCard onComplete={vi.fn()} />)
    fireEvent.click(screen.getByText('Get started'))
    expect(screen.getByText('Talk to me')).toBeDefined()
    fireEvent.click(screen.getByText('Open chat'))
    expect(screen.getByText('Tell me about yourself')).toBeDefined()
    fireEvent.click(screen.getByText('Add knowledge'))
    expect(screen.getByText('Shape my personality')).toBeDefined()
  })

  it('calls onComplete and sets localStorage on finish', () => {
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Get started'))
    fireEvent.click(screen.getByText('Open chat'))
    fireEvent.click(screen.getByText('Add knowledge'))
    fireEvent.click(screen.getByText('Customize me'))
    expect(onComplete).toHaveBeenCalled()
    expect(localStorage.setItem).toHaveBeenCalledWith('sloughgpt-onboarded', 'true')
  })

  it('skip button calls onComplete', () => {
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Skip'))
    expect(onComplete).toHaveBeenCalled()
  })
})
