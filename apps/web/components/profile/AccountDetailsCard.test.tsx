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
}))

import { AccountDetailsCard } from './AccountDetailsCard'

afterEach(() => cleanup())

describe('AccountDetailsCard', () => {
  it('renders the card title', () => {
    render(<AccountDetailsCard />)
    expect(screen.getAllByText('Account Details').length).toBeGreaterThanOrEqual(1)
  })

  it('renders labels', () => {
    render(<AccountDetailsCard />)
    expect(screen.getByText(/User ID/)).toBeTruthy()
    expect(screen.getByText(/Tenant ID/)).toBeTruthy()
    expect(screen.getByText(/Last Login/)).toBeTruthy()
    expect(screen.getByText(/Created/)).toBeTruthy()
  })

  it('shows dash placeholder when no data', () => {
    render(<AccountDetailsCard />)
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThanOrEqual(2)
  })

  it('displays provided user ID and tenant ID', () => {
    render(<AccountDetailsCard userId="abc-123" tenantId="t-456" />)
    expect(screen.getByText('abc-123')).toBeTruthy()
    expect(screen.getByText('t-456')).toBeTruthy()
  })

  it('formats date strings correctly', () => {
    const date = '2024-01-15T10:30:00Z'
    render(<AccountDetailsCard createdAt={date} />)
    expect(screen.getByText(new Date(date).toLocaleString())).toBeTruthy()
  })
})
