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

import { CompanionPresetCard } from './CompanionPresetCard'

afterEach(() => cleanup())

const mockPresets = [
  { id: 'warm', name: 'Warm', description: 'Friendly and supportive' },
  { id: 'curious', name: 'Curious', description: 'Asks great questions' },
  { id: 'professional', name: 'Professional', description: 'Direct and focused' },
]

describe('CompanionPresetCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<CompanionPresetCard presets={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders presets grid', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    expect(screen.getByText('Presets')).toBeTruthy()
    expect(screen.getByText('Warm')).toBeTruthy()
    expect(screen.getByText('Curious')).toBeTruthy()
    expect(screen.getByText('Professional')).toBeTruthy()
  })

  it('shows descriptions', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    expect(screen.getByText('Friendly and supportive')).toBeTruthy()
    expect(screen.getByText('Asks great questions')).toBeTruthy()
  })

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn()
    render(<CompanionPresetCard presets={mockPresets} onSelect={onSelect} />)
    fireEvent.click(screen.getByTestId('preset-warm'))
    expect(onSelect).toHaveBeenCalledWith('warm')
  })

  it('highlights active preset', () => {
    render(<CompanionPresetCard presets={mockPresets} activePreset="curious" />)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('selects and highlights on click', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    fireEvent.click(screen.getByTestId('preset-professional'))
    expect(screen.getByText('Active')).toBeTruthy()
  })
})
