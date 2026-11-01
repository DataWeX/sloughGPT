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

import { OnboardingTipsCard } from './OnboardingTipsCard'

afterEach(() => cleanup())

describe('OnboardingTipsCard', () => {
  it('renders tips', () => {
    render(<OnboardingTipsCard />)
    expect(screen.getByText('Tips & Tricks')).toBeTruthy()
    expect(screen.getByText('All')).toBeTruthy()
    expect(screen.getByText('Use /clear to reset')).toBeTruthy()
  })

  it('shows category filters', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const texts = buttons.map(b => b.textContent)
    expect(texts).toContain('Chat')
    expect(texts).toContain('Knowledge')
    expect(texts).toContain('General')
  })

  it('filters by category', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const chatBtn = buttons.find(b => b.textContent === 'Chat' && b.className.includes('text-[9px]'))
    fireEvent.click(chatBtn!)
    expect(screen.queryByText('Be specific')).toBeNull()
    expect(screen.getByText('Use /clear to reset')).toBeTruthy()
  })

  it('expands tip on click', () => {
    render(<OnboardingTipsCard />)
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.getByText(/Type \/clear in chat/)).toBeTruthy()
  })

  it('collapses on second click', () => {
    render(<OnboardingTipsCard />)
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.getByText(/Type \/clear in chat/)).toBeTruthy()
    fireEvent.click(screen.getByTestId('tip-t1'))
    expect(screen.queryByText(/Type \/clear in chat/)).toBeNull()
  })

  it('shows All filter selected by default', () => {
    render(<OnboardingTipsCard />)
    const buttons = screen.getAllByRole('button')
    const allBtn = buttons.find(b => b.textContent === 'All' && b.className.includes('text-[9px]'))
    expect(allBtn).toBeTruthy()
  })
})
