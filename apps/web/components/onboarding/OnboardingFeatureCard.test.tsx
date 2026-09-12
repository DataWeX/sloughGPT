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

import { OnboardingFeatureCard } from './OnboardingFeatureCard'

afterEach(() => cleanup())

describe('OnboardingFeatureCard', () => {
  it('renders feature grid', () => {
    render(<OnboardingFeatureCard />)
    expect(screen.getByText('Explore Features')).toBeTruthy()
    expect(screen.getByText('Smart Chat')).toBeTruthy()
    expect(screen.getByText('Knowledge Base')).toBeTruthy()
    expect(screen.getByText('Companion')).toBeTruthy()
  })

  it('shows all 8 features', () => {
    render(<OnboardingFeatureCard />)
    expect(screen.getByTestId('feature-chat')).toBeTruthy()
    expect(screen.getByTestId('feature-knowledge')).toBeTruthy()
    expect(screen.getByTestId('feature-companion')).toBeTruthy()
    expect(screen.getByTestId('feature-voice')).toBeTruthy()
    expect(screen.getByTestId('feature-images')).toBeTruthy()
    expect(screen.getByTestId('feature-files')).toBeTruthy()
    expect(screen.getByTestId('feature-analytics')).toBeTruthy()
    expect(screen.getByTestId('feature-training')).toBeTruthy()
  })

  it('expands feature on click', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Conversational AI that remembers context and learns from you.')).toBeTruthy()
  })

  it('collapses on second click', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Conversational AI that remembers context and learns from you.')).toBeTruthy()
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.queryByText('Conversational AI that remembers context and learns from you.')).toBeNull()
  })

  it('shows Open button when link exists', () => {
    render(<OnboardingFeatureCard />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    expect(screen.getByText('Open →')).toBeTruthy()
  })

  it('calls onNavigate', () => {
    const onNavigate = vi.fn()
    render(<OnboardingFeatureCard onNavigate={onNavigate} />)
    fireEvent.click(screen.getByTestId('feature-chat'))
    fireEvent.click(screen.getByText('Open →'))
    expect(onNavigate).toHaveBeenCalledWith('/chat')
  })
})
