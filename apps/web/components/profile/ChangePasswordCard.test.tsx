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

import { ChangePasswordCard } from './ChangePasswordCard'

afterEach(() => cleanup())

describe('ChangePasswordCard', () => {
  it('renders the card title', () => {
    render(<ChangePasswordCard />)
    expect(screen.getAllByText('Change Password').length).toBeGreaterThanOrEqual(1)
  })

  it('renders three password inputs', () => {
    const { container } = render(<ChangePasswordCard />)
    const inputs = container.querySelectorAll('input')
    expect(inputs.length).toBe(3)
  })

  it('renders the change password button', () => {
    render(<ChangePasswordCard />)
    const buttons = screen.getAllByText(/Change Password/)
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('disables button when fields are empty', () => {
    render(<ChangePasswordCard />)
    const btn = screen.getByRole('button')
    expect(btn).toHaveProperty('disabled', true)
  })

  it('shows changing state', () => {
    render(<ChangePasswordCard changingPassword />)
    expect(screen.getAllByText('Changing...').length).toBeGreaterThanOrEqual(1)
  })

  it('renders labels for each field', () => {
    render(<ChangePasswordCard />)
    expect(screen.getByText('Current Password')).toBeTruthy()
    expect(screen.getByText('New Password')).toBeTruthy()
    expect(screen.getByText('Confirm New Password')).toBeTruthy()
  })
})
