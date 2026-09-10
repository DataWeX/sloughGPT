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
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { AddMemberForm } from './AddMemberForm'

afterEach(() => { cleanup() })

describe('AddMemberForm', () => {
  it('renders title', () => {
    render(<AddMemberForm onAdd={() => {}} />)
    expect(screen.getByText('Add Member by ID')).toBeDefined()
  })

  it('renders user ID input', () => {
    render(<AddMemberForm onAdd={() => {}} />)
    expect(screen.getByPlaceholderText('User ID')).toBeDefined()
  })

  it('renders Add button', () => {
    render(<AddMemberForm onAdd={() => {}} />)
    expect(screen.getByText('Add')).toBeDefined()
  })

  it('renders role selector options', () => {
    render(<AddMemberForm onAdd={() => {}} />)
    expect(screen.getByText('Viewer')).toBeDefined()
    expect(screen.getByText('Member')).toBeDefined()
    expect(screen.getByText('Admin')).toBeDefined()
  })

  it('disables Add button when input is empty', () => {
    render(<AddMemberForm onAdd={() => {}} />)
    const btn = screen.getByText('Add')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })

  it('disables Add button when disabled prop is true', () => {
    render(<AddMemberForm onAdd={() => {}} disabled />)
    const btn = screen.getByText('Add')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })
})
