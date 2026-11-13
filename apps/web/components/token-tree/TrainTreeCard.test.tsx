import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TrainTreeCard } from './TrainTreeCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="vocab-input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,

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

const defaultProps = {
  vocabSize: 256,
  trainTexts: '',
  loading: false,
  onVocabSizeChange: vi.fn(),
  onTrainTextsChange: vi.fn(),
  onTrain: vi.fn(),
}

describe('TrainTreeCard', () => {
  it('renders the card title', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Train Tree')).toBeDefined()
  })

  it('renders the vocab size input', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByTestId('vocab-input')).toBeDefined()
  })

  it('displays the current vocab size value', () => {
    render(<TrainTreeCard {...defaultProps} vocabSize={512} />)
    expect((screen.getByTestId('vocab-input') as HTMLInputElement).value).toBe('512')
  })

  it('renders the train button with correct label', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Train Token Tree')).toBeDefined()
  })

  it('shows training state when loading', () => {
    render(<TrainTreeCard {...defaultProps} loading={true} />)
    expect(screen.getByText('Training...')).toBeDefined()
  })

  it('renders labels', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByText('Vocab Size')).toBeDefined()
  })

  it('renders the textarea for training texts', () => {
    render(<TrainTreeCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Leave empty to use default corpus...')).toBeDefined()
  })
})
