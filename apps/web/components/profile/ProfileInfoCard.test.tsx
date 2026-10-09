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
