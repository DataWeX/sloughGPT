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

import { CompanionPresetCard } from './CompanionPresetCard'

afterEach(() => cleanup())

const mockPresets = [
  { id: 'warm', name: 'Warm', description: 'Friendly and supportive' },
  { id: 'curious', name: 'Curious', description: 'Asks great questions' },
  { id: 'professional', name: 'Professional', description: 'Direct and focused' },
]

describe('CompanionPresetCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<CompanionPresetCard presets={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders presets grid', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    expect(screen.getByText('Presets')).toBeTruthy()
    expect(screen.getByText('Warm')).toBeTruthy()
    expect(screen.getByText('Curious')).toBeTruthy()
    expect(screen.getByText('Professional')).toBeTruthy()
  })

  it('shows descriptions', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    expect(screen.getByText('Friendly and supportive')).toBeTruthy()
    expect(screen.getByText('Asks great questions')).toBeTruthy()
  })

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn()
    render(<CompanionPresetCard presets={mockPresets} onSelect={onSelect} />)
    fireEvent.click(screen.getByTestId('preset-warm'))
    expect(onSelect).toHaveBeenCalledWith('warm')
  })

  it('highlights active preset', () => {
    render(<CompanionPresetCard presets={mockPresets} activePreset="curious" />)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('selects and highlights on click', () => {
    render(<CompanionPresetCard presets={mockPresets} />)
    fireEvent.click(screen.getByTestId('preset-professional'))
    expect(screen.getByText('Active')).toBeTruthy()
  })
})
