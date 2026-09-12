/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityCoreIdentity } from './PersonalityCoreIdentity'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Input: ({ value, onChange, placeholder, ...props }: any) => (
    <input data-testid="input" value={value} onChange={onChange} placeholder={placeholder} {...props} />
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

afterEach(() => cleanup())

describe('PersonalityCoreIdentity', () => {
  const defaultProps = {
    values: 'honesty, curiosity',
    goals: 'help users',
    interests: 'AI, science',
    avoid: 'being rude',
    onValuesChange: vi.fn(),
    onGoalsChange: vi.fn(),
    onInterestsChange: vi.fn(),
    onAvoidChange: vi.fn(),
  }

  it('renders title "Core Identity"', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByText('Core Identity')).toBeDefined()
  })

  it('renders four input fields', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs).toHaveLength(4)
  })

  it('displays current values', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect((inputs[0] as HTMLInputElement).value).toBe('honesty, curiosity')
    expect((inputs[1] as HTMLInputElement).value).toBe('help users')
  })

  it('renders labels for each field', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByText('Values (comma-separated)')).toBeDefined()
    expect(screen.getByText('Goals (comma-separated)')).toBeDefined()
    expect(screen.getByText('Interests (comma-separated)')).toBeDefined()
    expect(screen.getByText('Avoid (comma-separated)')).toBeDefined()
  })

  it('renders placeholders', () => {
    render(<PersonalityCoreIdentity {...defaultProps} />)
    expect(screen.getByPlaceholderText('helpfulness, honesty, curiosity')).toBeDefined()
    expect(screen.getByPlaceholderText('AI, programming, science')).toBeDefined()
  })
})
