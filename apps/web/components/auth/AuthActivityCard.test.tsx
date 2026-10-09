// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '5m ago',
}))

import { AuthActivityCard, recordAuthEvent } from './AuthActivityCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('AuthActivityCard', () => {
  it('shows empty state', () => {
    render(<AuthActivityCard />)
    expect(screen.getByText('Activity')).toBeTruthy()
    expect(screen.getByText('No activity recorded.')).toBeTruthy()
  })

  it('shows recorded events', () => {
    recordAuthEvent('login', '192.168.1.1')
    recordAuthEvent('logout')
    render(<AuthActivityCard />)
    expect(screen.getByText('(2)')).toBeTruthy()
    expect(screen.getByText('login')).toBeTruthy()
    expect(screen.getByText('logout')).toBeTruthy()
    expect(screen.getByText('192.168.1.1')).toBeTruthy()
  })

  it('filters by type', () => {
    recordAuthEvent('login')
    recordAuthEvent('register')
    recordAuthEvent('login')
    render(<AuthActivityCard />)
    const loginBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('login'))
    if (loginBtn) fireEvent.click(loginBtn)
    expect(screen.queryByText('register')).toBeNull()
  })

  it('clears activity', () => {
    recordAuthEvent('login')
    recordAuthEvent('logout')
    render(<AuthActivityCard />)
    fireEvent.click(screen.getByText('Clear'))
    expect(screen.getByText('No activity recorded.')).toBeTruthy()
  })

  it('loads from localStorage', () => {
    localStorage.setItem('sloughgpt-auth-activity', JSON.stringify([
      { id: 'evt-1', type: 'failed_login', timestamp: Date.now(), details: 'Wrong password' },
    ]))
    render(<AuthActivityCard />)
    expect(screen.getByText('failed login')).toBeTruthy()
    expect(screen.getByText('Wrong password')).toBeTruthy()
  })
})
