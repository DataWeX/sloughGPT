/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityVoiceSliders } from './PersonalityVoiceSliders'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

afterEach(() => cleanup())

const mockVoice = {
  formality: 0.5,
  warmth: 0.8,
  confidence: 0.7,
  humor: 0.3,
  verbosity: 0.6,
  empathy: 0.9,
}

describe('PersonalityVoiceSliders', () => {
  it('renders title "Voice"', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('Voice')).toBeDefined()
  })

  it('renders six voice labels', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('Formality')).toBeDefined()
    expect(screen.getByText('Warmth')).toBeDefined()
    expect(screen.getByText('Confidence')).toBeDefined()
    expect(screen.getByText('Humor')).toBeDefined()
    expect(screen.getByText('Verbosity')).toBeDefined()
    expect(screen.getByText('Empathy')).toBeDefined()
  })

  it('renders six range inputs', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    const sliders = screen.getAllByRole('slider')
    expect(sliders).toHaveLength(6)
  })

  it('displays percentage values', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('50%')).toBeDefined()
    expect(screen.getByText('80%')).toBeDefined()
    expect(screen.getByText('90%')).toBeDefined()
  })

  it('renders card description', () => {
    render(<PersonalityVoiceSliders voice={mockVoice} onVoiceChange={vi.fn()} />)
    expect(screen.getByText('How the system sounds when communicating')).toBeDefined()
  })
})
