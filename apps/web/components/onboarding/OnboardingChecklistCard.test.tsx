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

import { OnboardingChecklistCard } from './OnboardingChecklistCard'

afterEach(() => { cleanup(); localStorage.clear() })

const ALL_IDS = ['first-chat', 'add-knowledge', 'customize-companion', 'try-voice', 'upload-file', 'generate-image', 'check-analytics', 'explore-settings']

describe('OnboardingChecklistCard', () => {
  it('renders checklist', () => {
    render(<OnboardingChecklistCard />)
    expect(screen.getByText('Getting Started')).toBeTruthy()
    expect(screen.getByText(/0\/8/)).toBeTruthy()
    expect(screen.getByText('Send your first message')).toBeTruthy()
    expect(screen.getByText('Add a knowledge entry')).toBeTruthy()
  })

  it('shows progress bar', () => {
    render(<OnboardingChecklistCard />)
    expect(screen.getByTestId('onboarding-checklist').querySelector('[style*="width"]')).toBeTruthy()
  })

  it('toggles completion', () => {
    render(<OnboardingChecklistCard />)
    fireEvent.click(screen.getByTestId('checklist-first-chat'))
    expect(screen.getByText(/1\/8/)).toBeTruthy()
  })

  it('persists to localStorage', () => {
    render(<OnboardingChecklistCard />)
    fireEvent.click(screen.getByTestId('checklist-first-chat'))
    const stored = JSON.parse(localStorage.getItem('sloughgpt-onboarding-checklist')!)
    expect(stored['first-chat']).toBe(true)
  })

  it('loads from localStorage', () => {
    localStorage.setItem('sloughgpt-onboarding-checklist', JSON.stringify({ 'first-chat': true, 'add-knowledge': true }))
    render(<OnboardingChecklistCard />)
    expect(screen.getByText(/2\/8/)).toBeTruthy()
  })

  it('hides when all complete', () => {
    const allDone: Record<string, boolean> = {}
    ALL_IDS.forEach(id => { allDone[id] = true })
    localStorage.setItem('sloughgpt-onboarding-checklist', JSON.stringify(allDone))
    const { container } = render(<OnboardingChecklistCard />)
    expect(container.firstChild).toBeNull()
  })
})
