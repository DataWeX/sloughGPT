// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
}))

import { SettingsMemoryCard } from './SettingsMemoryCard'

afterEach(() => cleanup())

describe('SettingsMemoryCard', () => {
  it('renders title and description', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByText('Memory')).toBeTruthy()
    expect(screen.getByText('Custom instructions included with every prompt')).toBeTruthy()
  })

  it('renders textarea with custom context', () => {
    render(<SettingsMemoryCard customContext="You are helpful" onChange={() => {}} />)
    expect(screen.getByDisplayValue('You are helpful')).toBeTruthy()
  })

  it('renders textarea with empty value', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByDisplayValue('')).toBeTruthy()
  })

  it('calls onChange when textarea updated', () => {
    const onChange = vi.fn()
    render(<SettingsMemoryCard customContext="" onChange={onChange} />)
    fireEvent.change(screen.getByRole('textbox', { name: /custom instructions/i }), { target: { value: 'New text' } })
    expect(onChange).toHaveBeenCalledWith('New text')
  })

  it('has correct aria-label on textarea', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByRole('textbox', { name: /custom instructions/i })).toBeTruthy()
  })
})
