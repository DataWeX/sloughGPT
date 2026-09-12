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

import { FeedbackFormCard } from './FeedbackFormCard'

afterEach(() => { cleanup() })

describe('FeedbackFormCard', () => {
  it('renders the form', () => {
    render(<FeedbackFormCard />)
    expect(screen.getAllByText('Submit Feedback').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('👍 Good')).toBeTruthy()
    expect(screen.getByText('👎 Bad')).toBeTruthy()
  })

  it('submit button disabled without rating', () => {
    render(<FeedbackFormCard />)
    const btn = screen.getByRole('button', { name: /Submit Feedback/ })
    expect(btn).toHaveProperty('disabled', true)
  })

  it('enables submit after selecting rating', () => {
    render(<FeedbackFormCard />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    const btn = screen.getByRole('button', { name: /Submit Feedback/ })
    expect(btn).not.toHaveProperty('disabled', true)
  })

  it('calls onSubmit with rating', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ rating: 'thumbs_up' }))
  })

  it('includes comment when provided', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs down'))
    fireEvent.change(screen.getByLabelText('Feedback comment'), { target: { value: 'Needs improvement' } })
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ comment: 'Needs improvement' }))
  })

  it('includes category when selected', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    fireEvent.click(screen.getByText('quality'))
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ category: 'quality' }))
  })

  it('shows categories', () => {
    render(<FeedbackFormCard />)
    expect(screen.getByText('quality')).toBeTruthy()
    expect(screen.getByText('accuracy')).toBeTruthy()
    expect(screen.getByText('helpfulness')).toBeTruthy()
    expect(screen.getByText('speed')).toBeTruthy()
    expect(screen.getByText('other')).toBeTruthy()
  })

  it('shows submitting state', () => {
    render(<FeedbackFormCard submitting />)
    const btn = screen.getByRole('button', { name: /Submitting/ })
    expect(btn).toHaveProperty('disabled', true)
  })
})
