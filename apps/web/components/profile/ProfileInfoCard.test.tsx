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
  Input: (props: any) => <input {...props} />,
}))

import { ProfileInfoCard } from './ProfileInfoCard'

afterEach(() => cleanup())

describe('ProfileInfoCard', () => {
  it('renders the card title', () => {
    render(<ProfileInfoCard />)
    expect(screen.getAllByText('Profile Information').length).toBeGreaterThanOrEqual(1)
  })

  it('renders username input as disabled', () => {
    render(<ProfileInfoCard username="testuser" />)
    const inputs = screen.getAllByRole('textbox')
    const disabledInput = inputs.find((i) => i.hasAttribute('disabled'))
    expect(disabledInput).toBeTruthy()
    expect(disabledInput?.getAttribute('value')).toBe('testuser')
  })

  it('renders display name and email inputs', () => {
    render(<ProfileInfoCard />)
    const inputs = screen.getAllByRole('textbox')
    expect(inputs.length).toBe(3)
  })

  it('shows save button', () => {
    render(<ProfileInfoCard />)
    expect(screen.getAllByText('Save Changes').length).toBeGreaterThanOrEqual(1)
  })

  it('shows saving state', () => {
    render(<ProfileInfoCard saving />)
    expect(screen.getAllByText('Saving...').length).toBeGreaterThanOrEqual(1)
  })

  it('renders with provided data', () => {
    render(<ProfileInfoCard username="alice" displayName="Alice B" email="alice@test.com" />)
    const inputs = screen.getAllByRole('textbox')
    expect(inputs[0].getAttribute('value')).toBe('alice')
    expect(inputs[1].getAttribute('value')).toBe('Alice B')
    expect(inputs[2].getAttribute('value')).toBe('alice@test.com')
  })
})
