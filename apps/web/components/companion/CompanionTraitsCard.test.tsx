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

import { CompanionTraitsCard } from './CompanionTraitsCard'

afterEach(() => cleanup())

const mockTraits = { name: 'Test', warmth: 0.8, curiosity: 0.6, creativity: 0.4, confidence: 0.7, humor: 0.3 }

describe('CompanionTraitsCard', () => {
  it('renders empty state', () => {
    render(<CompanionTraitsCard traits={null} />)
    expect(screen.getByText('No traits loaded.')).toBeTruthy()
  })

  it('shows trait sliders', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('Personality Traits')).toBeTruthy()
    expect(screen.getByTestId('trait-warmth')).toBeTruthy()
    expect(screen.getByTestId('trait-curiosity')).toBeTruthy()
    expect(screen.getByTestId('trait-creativity')).toBeTruthy()
    expect(screen.getByTestId('trait-confidence')).toBeTruthy()
    expect(screen.getByTestId('trait-humor')).toBeTruthy()
  })

  it('shows percentage labels', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('80%')).toBeTruthy()
    expect(screen.getByText('60%')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
  })

  it('shows overall average bar', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('Overall')).toBeTruthy()
  })

  it('shows Save button when draft changes', () => {
    render(<CompanionTraitsCard traits={mockTraits} onSave={vi.fn()} />)
    const slider = screen.getByTestId('trait-warmth')
    fireEvent.change(slider, { target: { value: 0.9 } })
    expect(screen.getByText('Save')).toBeTruthy()
  })

  it('calls onSave with updated traits', () => {
    const onSave = vi.fn()
    render(<CompanionTraitsCard traits={mockTraits} onSave={onSave} />)
    const slider = screen.getByTestId('trait-warmth')
    fireEvent.change(slider, { target: { value: 0.95 } })
    fireEvent.click(screen.getByText('Save'))
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ warmth: 0.95 }))
  })

  it('shows Reset button when onReset provided', () => {
    render(<CompanionTraitsCard traits={mockTraits} onReset={vi.fn()} />)
    expect(screen.getByText('Reset')).toBeTruthy()
  })
})
