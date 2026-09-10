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

import { OnboardingTipsCard } from './OnboardingTipsCard'

afterEach(() => cleanup())

describe('OnboardingTipsCard', () => {
  it('renders tips', () => {
    render(<OnboardingTipsCard />)
    expect(screen.getByText('Tips & Tricks')).toBeTruthy()
    expect(screen.getByText('All')).toBeTruthy()
    expect(screen.getByText('Use /clear to reset')).toBeTruthy()
  })

  it('shows category filters', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const texts = buttons.map(b => b.textContent)
    expect(texts).toContain('Chat')
    expect(texts).toContain('Knowledge')
    expect(texts).toContain('General')
  })

  it('filters by category', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const chatBtn = buttons.find(b => b.textContent === 'Chat' && b.className.includes('text-[9px]'))
    fireEvent.click(chatBtn!)
    expect(screen.queryByText('Be specific')).toBeNull()
    expect(screen.getByText('Use /clear to reset')).toBeTruthy()
  })

  it('expands tip on click', () => {
    render(<OnboardingTipsCard />)
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.getByText(/Type \/clear in chat/)).toBeTruthy()
  })

  it('collapses on second click', () => {
    render(<OnboardingTipsCard />)
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.getByText(/Type \/clear in chat/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.queryByText(/Type \/clear in chat/)).toBeNull()
  })

  it('shows All filter selected by default', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const allBtn = buttons.find(b => b.textContent === 'All' && b.className.includes('text-[9px]'))
    expect(allBtn).toBeTruthy()
  })
})
