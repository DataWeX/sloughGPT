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
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { ApiKeyCreateCard } from './ApiKeyCreateCard'

afterEach(() => cleanup())

describe('ApiKeyCreateCard', () => {
  it('renders create form', () => {
    render(<ApiKeyCreateCard />)
    expect(screen.getByText('Create API Key')).toBeTruthy()
    expect(screen.getByPlaceholderText('Key name')).toBeTruthy()
    expect(screen.getByText('Create')).toBeTruthy()
  })

  it('disables create when empty', () => {
    render(<ApiKeyCreateCard />)
    expect(screen.getByText('Create').hasAttribute('disabled')).toBe(true)
  })

  it('enables create when name entered', () => {
    render(<ApiKeyCreateCard />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'my-key' } })
    expect(screen.getByText('Create').hasAttribute('disabled')).toBe(false)
  })

  it('calls onCreate', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'test-key' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith('test-key')
    })
  })

  it('clears name after create', async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined)
    render(<ApiKeyCreateCard onCreate={onCreate} />)
    fireEvent.change(screen.getByPlaceholderText('Key name'), { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Key name')).toHaveValue('')
    })
  })
})
