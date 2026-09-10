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

import { WorkspaceSelector } from './WorkspaceSelector'

afterEach(() => { cleanup() })

const workspaces = [
  { id: 'ws-1', name: 'Acme Corp', member_count: 5 },
  { id: 'ws-2', name: 'Beta Inc', member_count: 12 },
]

describe('WorkspaceSelector', () => {
  it('renders title', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('Select Workspace')).toBeDefined()
  })

  it('renders all workspace buttons', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText(/Acme Corp/)).toBeDefined()
    expect(screen.getByText(/Beta Inc/)).toBeDefined()
  })

  it('displays member counts', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('(5)')).toBeDefined()
    expect(screen.getByText('(12)')).toBeDefined()
  })

  it('calls onSelect when workspace is clicked', () => {
    const onSelect = vi.fn()
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={onSelect} />)
    screen.getByText(/Acme Corp/).click()
    expect(onSelect).toHaveBeenCalledWith('ws-1')
  })

  it('highlights the selected workspace', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId="ws-2" onSelect={() => {}} />)
    const btn = screen.getByText(/Beta Inc/)
    expect(btn.className).toContain('bg-primary')
  })

  it('renders empty list without error', () => {
    render(<WorkspaceSelector workspaces={[]} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('Select Workspace')).toBeDefined()
  })
})
