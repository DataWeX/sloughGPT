// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, ...p }: any) => <button onClick={onClick} {...p}>{children}</button>,

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

import { OnboardingCard } from './OnboardingCard'

beforeEach(() => {
  vi.clearAllMocks()
  Storage.prototype.getItem = vi.fn(() => null)
  Storage.prototype.setItem = vi.fn()
})

afterEach(() => cleanup())

describe('OnboardingCard', () => {
  it('renders welcome step', () => {
    render(<OnboardingCard onComplete={vi.fn()} />)
    expect(screen.getByText('Welcome to SloughGPT')).toBeDefined()
  })

  it('calls onComplete immediately if already onboarded', () => {
    Storage.prototype.getItem = vi.fn(() => 'true')
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    expect(onComplete).toHaveBeenCalled()
  })

  it('advances through all steps', () => {
    render(<OnboardingCard onComplete={vi.fn()} />)
    fireEvent.click(screen.getByText('Get started'))
    expect(screen.getByText('Talk to me')).toBeDefined()
    fireEvent.click(screen.getByText('Open chat'))
    expect(screen.getByText('Tell me about yourself')).toBeDefined()
    fireEvent.click(screen.getByText('Add knowledge'))
    expect(screen.getByText('Shape my personality')).toBeDefined()
  })

  it('calls onComplete and sets localStorage on finish', () => {
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Get started'))
    fireEvent.click(screen.getByText('Open chat'))
    fireEvent.click(screen.getByText('Add knowledge'))
    fireEvent.click(screen.getByText('Customize me'))
    expect(onComplete).toHaveBeenCalled()
    expect(localStorage.setItem).toHaveBeenCalledWith('sloughgpt-onboarded', 'true')
  })

  it('skip button calls onComplete', () => {
    const onComplete = vi.fn()
    render(<OnboardingCard onComplete={onComplete} />)
    fireEvent.click(screen.getByText('Skip'))
    expect(onComplete).toHaveBeenCalled()
  })
})
