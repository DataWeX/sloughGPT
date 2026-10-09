/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorldRenderConfigCard } from './WorldRenderConfigCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="config-input" {...props} />,
  Label: ({ children, ...props }: any) => <label {...props}>{children}</label>,

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

describe('WorldRenderConfigCard', () => {
  const defaultProps = {
    config: { width: 160, height: 120, samples: 16, camera_height: 40, camera_distance: 30 },
    onConfigChange: vi.fn(),
    onRender: vi.fn(),
  }

  it('renders the title', () => {
    render(<WorldRenderConfigCard {...defaultProps} />)
    expect(screen.getByText('Render Config')).toBeDefined()
  })

  it('renders the Render button', () => {
    render(<WorldRenderConfigCard {...defaultProps} />)
    expect(screen.getByText('Render')).toBeDefined()
  })

  it('calls onRender when Render is clicked', () => {
    const onRender = vi.fn()
    render(<WorldRenderConfigCard {...defaultProps} onRender={onRender} />)
    fireEvent.click(screen.getByText('Render'))
    expect(onRender).toHaveBeenCalledOnce()
  })

  it('renders all 5 config inputs', () => {
    render(<WorldRenderConfigCard {...defaultProps} />)
    expect(screen.getAllByTestId('config-input')).toHaveLength(5)
  })

  it('renders tick buttons when onTick provided', () => {
    render(<WorldRenderConfigCard {...defaultProps} onTick={vi.fn()} />)
    expect(screen.getByText('Run Tick')).toBeDefined()
    expect(screen.getByText('Tick + Neural')).toBeDefined()
  })

  it('does not render tick buttons when onTick not provided', () => {
    render(<WorldRenderConfigCard {...defaultProps} />)
    expect(screen.queryByText('Run Tick')).toBeNull()
  })

  it('shows rendering state', () => {
    render(<WorldRenderConfigCard {...defaultProps} rendering />)
    expect(screen.getByText('Rendering...')).toBeDefined()
  })

  it('shows ticking state', () => {
    render(<WorldRenderConfigCard {...defaultProps} onTick={vi.fn()} ticking />)
    expect(screen.getByText('Ticking...')).toBeDefined()
    expect(screen.getByText('Processing...')).toBeDefined()
  })
})
