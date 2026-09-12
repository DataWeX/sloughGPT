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

import { WeightsTestCard } from './WeightsTestCard'

afterEach(() => { cleanup() })

describe('WeightsTestCard', () => {
  it('renders title', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Test Weights')).toBeDefined()
  })

  it('renders message input', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByPlaceholderText('Type a message to test...')).toBeDefined()
  })

  it('renders Compute button', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Compute')).toBeDefined()
  })

  it('disables Compute when message is empty', () => {
    render(<WeightsTestCard testMessage="" onMessageChange={() => {}} onCompute={() => {}} />)
    expect(screen.getByText('Compute').hasAttribute('disabled')).toBe(true)
  })

  it('shows computing state', () => {
    render(<WeightsTestCard testMessage="hello" onMessageChange={() => {}} onCompute={() => {}} testing />)
    expect(screen.getByText('Computing...')).toBeDefined()
  })

  it('renders weight bars when weights provided', () => {
    render(
      <WeightsTestCard
        testMessage="test"
        onMessageChange={() => {}}
        onCompute={() => {}}
        weights={{
          temperature: 0.8,
          top_p: 0.9,
          repetition_penalty: 1.2,
          style_bias: 0.5,
          confidence_boost: 0.7,
          top_k: 40,
          based_on_samples: 15,
        }}
      />
    )
    expect(screen.getByText('Temperature')).toBeDefined()
    expect(screen.getByText('Top P')).toBeDefined()
    expect(screen.getByText('Repetition Penalty')).toBeDefined()
    expect(screen.getByText('Style Bias')).toBeDefined()
    expect(screen.getByText('Confidence Boost')).toBeDefined()
    expect(screen.getByText('Top K')).toBeDefined()
  })

  it('renders sample count', () => {
    render(
      <WeightsTestCard
        testMessage="test"
        onMessageChange={() => {}}
        onCompute={() => {}}
        weights={{
          temperature: 0.8,
          top_p: 0.9,
          repetition_penalty: 1.2,
          style_bias: 0.5,
          confidence_boost: 0.7,
          top_k: 40,
          based_on_samples: 15,
        }}
      />
    )
    expect(screen.getByText('Based on 15 feedback samples')).toBeDefined()
  })
})
