// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { OnboardingFeatureCard } from './OnboardingFeatureCard'

afterEach(() => cleanup())

describe('OnboardingFeatureCard', () => {
  it('renders feature grid', () => {
    render(<OnboardingFeatureCard />)
    expect(screen.getByText('Explore Features')).toBeTruthy()
    expect(screen.getByText('Smart Chat')).toBeTruthy()
    expect(screen.getByText('Knowledge Base')).toBeTruthy()
    expect(screen.getByText('Companion')).toBeTruthy()
  })

  it('shows all 8 features', () => {
    render(<OnboardingFeatureCard />)
    expect(screen.getByTestId('feature-chat')).toBeTruthy()
    expect(screen.getByTestId('feature-knowledge')).toBeTruthy()
    expect(screen.getByTestId('feature-companion')).toBeTruthy()
    expect(screen.getByTestId('feature-voice')).toBeTruthy()
    expect(screen.getByTestId('feature-images')).toBeTruthy()
    expect(screen.getByTestId('feature-files')).toBeTruthy()
    expect(screen.getByTestId('feature-analytics')).toBeTruthy()
    expect(screen.getByTestId('feature-training')).toBeTruthy()
  })

  it('expands feature on click', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Conversational AI that remembers context and learns from you.')).toBeTruthy()
  })

  it('collapses on second click', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Conversational AI that remembers context and learns from you.')).toBeTruthy()
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.queryByText('Conversational AI that remembers context and learns from you.')).toBeNull()
  })

  it('shows Open button when link exists', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Open →')).toBeTruthy()
  })

  it('calls onNavigate', () => {
    const onNavigate = vi.fn()
    render(<OnboardingFeatureCard onNavigate={onNavigate} />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    fireEvent.click(screen.getByText('Open →'))
    expect(onNavigate).toHaveBeenCalledWith('/chat')
  })
})
