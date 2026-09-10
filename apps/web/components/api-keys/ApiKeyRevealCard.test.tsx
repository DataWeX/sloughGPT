// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

Object.defineProperty(navigator, 'clipboard', {
  value: { writeText: vi.fn().mockResolvedValue(undefined) },
})

import { ApiKeyRevealCard } from './ApiKeyRevealCard'

afterEach(() => cleanup())

describe('ApiKeyRevealCard', () => {
  it('renders nothing when no key', () => {
    const { container } = render(<ApiKeyRevealCard revealedKey={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows new key', () => {
    render(<ApiKeyRevealCard revealedKey="sk-secret-key-123" />)
    expect(screen.getByText(/New API Key/)).toBeTruthy()
    expect(screen.getByText(/sk-secret-key-123/)).toBeTruthy()
  })

  it('shows copy and dismiss buttons', () => {
    const onDismiss = vi.fn()
    render(<ApiKeyRevealCard revealedKey="sk-abc" onDismiss={onDismiss} />)
    const btns = screen.getAllByRole('button')
    expect(btns.some(b => b.textContent === 'Copy')).toBe(true)
    expect(btns.some(b => b.textContent === 'Dismiss')).toBe(true)
  })

  it('copies key to clipboard', async () => {
    render(<ApiKeyRevealCard revealedKey="sk-abc" />)
    const copyBtn = screen.getAllByRole('button').find(b => b.textContent === 'Copy')!
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('sk-abc')
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeTruthy()
    })
  })

  it('calls onDismiss', () => {
    const onDismiss = vi.fn()
    render(<ApiKeyRevealCard revealedKey="sk-abc" onDismiss={onDismiss} />)
    const dismissBtn = screen.getAllByRole('button').find(b => b.textContent === 'Dismiss')!
    fireEvent.click(dismissBtn)
    expect(onDismiss).toHaveBeenCalled()
  })
})
