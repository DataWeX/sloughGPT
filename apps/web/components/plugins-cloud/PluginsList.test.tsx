// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Skeleton: (props: any) => <div data-testid="skeleton" {...props} />,
}))

import { PluginsList } from './PluginsList'
import type { PluginInfo } from './PluginsList'

afterEach(() => cleanup())

const mockPlugins: PluginInfo[] = [
  { name: 'my-plugin', version: '1.0.0', description: 'A plugin', author: 'Alice', enabled: true },
  { name: 'other-plugin', version: '2.1.0', description: 'Another', author: 'Bob', enabled: false },
]

describe('PluginsList', () => {
  it('renders the card title', () => {
    render(<PluginsList />)
    expect(screen.getAllByText('Installed Plugins').length).toBeGreaterThanOrEqual(1)
  })

  it('renders the card description', () => {
    render(<PluginsList />)
    expect(screen.getAllByText('Manage loaded plugins').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no plugins', () => {
    render(<PluginsList />)
    expect(screen.getByText(/No plugins installed/)).toBeTruthy()
  })

  it('shows skeleton when loading', () => {
    render(<PluginsList loading />)
    expect(screen.getByTestId('skeleton')).toBeTruthy()
  })

  it('renders plugins when provided', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('my-plugin')).toBeTruthy()
    expect(screen.getByText('other-plugin')).toBeTruthy()
  })

  it('renders plugin versions and authors', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('v1.0.0')).toBeTruthy()
    expect(screen.getByText('v2.1.0')).toBeTruthy()
    expect(screen.getByText('by Alice')).toBeTruthy()
    expect(screen.getByText('by Bob')).toBeTruthy()
  })

  it('renders correct badge text', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('Enabled')).toBeTruthy()
    expect(screen.getByText('Disabled')).toBeTruthy()
  })
})
