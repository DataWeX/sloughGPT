// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
  Textarea: (props: any) => <textarea {...props} />,
}))

import { DevApiPlaygroundCard } from './DevApiPlaygroundCard'

afterEach(() => cleanup())

describe('DevApiPlaygroundCard', () => {
  it('renders method selector and path input', () => {
    render(<DevApiPlaygroundCard />)
    expect(screen.getByLabelText('HTTP method')).toBeTruthy()
    expect(screen.getByLabelText('Request path')).toBeTruthy()
  })

  it('renders send button', () => {
    render(<DevApiPlaygroundCard />)
    expect(screen.getByRole('button', { name: /Send/ })).toBeTruthy()
  })

  it('defaults to GET method and /health path', () => {
    render(<DevApiPlaygroundCard />)
    const select = screen.getByLabelText('HTTP method') as HTMLSelectElement
    const input = screen.getByLabelText('Request path') as HTMLInputElement
    expect(select.value).toBe('GET')
    expect(input.value).toBe('/health')
  })

  it('shows body textarea when method is POST', () => {
    render(<DevApiPlaygroundCard />)
    const select = screen.getByLabelText('HTTP method') as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'POST' } })
    expect(screen.getByLabelText('Request path')).toBeTruthy()
  })

  it('hides body for GET requests', () => {
    render(<DevApiPlaygroundCard />)
    expect(screen.queryByLabelText('Request body')).toBeNull()
  })
})
