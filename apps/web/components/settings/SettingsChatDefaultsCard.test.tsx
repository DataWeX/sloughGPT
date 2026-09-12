// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Slider: ({ value, onValueChange, ...props }: any) => (
    <input type="range" value={value?.[0] ?? 0} onChange={(e) => onValueChange?.([Number(e.target.value)])} {...props} />
  ),
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={(e) => onCheckedChange?.(e.target.checked)} {...props} />
  ),

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

import { SettingsChatDefaultsCard } from './SettingsChatDefaultsCard'

afterEach(() => cleanup())

const defaultProps = {
  temperature: 0.7,
  maxTokens: 512,
  topP: 0.9,
  topK: 50,
  streaming: true,
  collapsibleMessageLength: 200,
  onTemperatureChange: vi.fn(),
  onMaxTokensChange: vi.fn(),
  onTopPChange: vi.fn(),
  onTopKChange: vi.fn(),
  onStreamingChange: vi.fn(),
  onCollapsibleMessageLengthChange: vi.fn(),
  onReset: vi.fn(),
  version: '3.0.0',
}

describe('SettingsChatDefaultsCard', () => {
  it('renders title and description', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Chat defaults')).toBeTruthy()
    expect(screen.getByText('Default model and generation settings')).toBeTruthy()
  })

  it('displays temperature slider value', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Temperature')).toBeTruthy()
    expect(screen.getByText('0.7')).toBeTruthy()
  })

  it('displays streaming toggle', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Streaming')).toBeTruthy()
    expect(screen.getByText('Show tokens as they are generated')).toBeTruthy()
  })

  it('calls onStreamingChange when streaming toggled', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    const toggle = screen.getByRole('checkbox', { name: /toggle streaming/i })
    fireEvent.click(toggle)
    expect(defaultProps.onStreamingChange).toHaveBeenCalledWith(false)
  })

  it('calls onReset when reset clicked', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    fireEvent.click(screen.getByText('Reset'))
    expect(defaultProps.onReset).toHaveBeenCalledOnce()
  })

  it('shows collapsible message length label', () => {
    render(<SettingsChatDefaultsCard {...defaultProps} />)
    expect(screen.getByText('Auto-collapse messages longer than')).toBeTruthy()
  })
})
