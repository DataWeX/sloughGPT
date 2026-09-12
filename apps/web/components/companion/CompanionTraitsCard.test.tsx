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

import { CompanionTraitsCard } from './CompanionTraitsCard'

afterEach(() => cleanup())

const mockTraits = { name: 'Test', warmth: 0.8, curiosity: 0.6, creativity: 0.4, confidence: 0.7, humor: 0.3 }

describe('CompanionTraitsCard', () => {
  it('renders empty state', () => {
    render(<CompanionTraitsCard traits={null} />)
    expect(screen.getByText('No traits loaded.')).toBeTruthy()
  })

  it('shows trait sliders', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('Personality Traits')).toBeTruthy()
    expect(screen.getByTestId('trait-warmth')).toBeTruthy()
    expect(screen.getByTestId('trait-curiosity')).toBeTruthy()
    expect(screen.getByTestId('trait-creativity')).toBeTruthy()
    expect(screen.getByTestId('trait-confidence')).toBeTruthy()
    expect(screen.getByTestId('trait-humor')).toBeTruthy()
  })

  it('shows percentage labels', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('80%')).toBeTruthy()
    expect(screen.getByText('60%')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
  })

  it('shows overall average bar', () => {
    render(<CompanionTraitsCard traits={mockTraits} />)
    expect(screen.getByText('Overall')).toBeTruthy()
  })

  it('shows Save button when draft changes', () => {
    render(<CompanionTraitsCard traits={mockTraits} onSave={vi.fn()} />)
    const slider = screen.getByTestId('trait-warmth')
    fireEvent.change(slider, { target: { value: 0.9 } })
    expect(screen.getByText('Save')).toBeTruthy()
  })

  it('calls onSave with updated traits', () => {
    const onSave = vi.fn()
    render(<CompanionTraitsCard traits={mockTraits} onSave={onSave} />)
    const slider = screen.getByTestId('trait-warmth')
    fireEvent.change(slider, { target: { value: 0.95 } })
    fireEvent.click(screen.getByText('Save'))
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ warmth: 0.95 }))
  })

  it('shows Reset button when onReset provided', () => {
    render(<CompanionTraitsCard traits={mockTraits} onReset={vi.fn()} />)
    expect(screen.getByText('Reset')).toBeTruthy()
  })
})
