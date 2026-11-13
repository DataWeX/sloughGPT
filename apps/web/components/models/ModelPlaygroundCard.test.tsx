import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockGenerateStream = vi.fn()

vi.mock('@/lib/generate-controller', () => ({
  generateController: {
    generate: vi.fn(),
    generateStream: (...args: any[]) => mockGenerateStream(...args),
  },
}))
vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Slider: ({ label, value, onValueChange, min, max, step }: any) => (
      <input
        data-testid={`slider-${label}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value?.[0] ?? 0}
        onChange={e => onValueChange?.([parseFloat(e.target.value)])}
        aria-label={label}
      />
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
}
})

import ModelPlaygroundCard from './ModelPlaygroundCard'

describe('ModelPlaygroundCard', () => {
  afterEach(cleanup)
  beforeEach(() => vi.clearAllMocks())

  it('renders null when no activeRuntimeId', () => {
    const { container } = render(<ModelPlaygroundCard activeRuntimeId={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders card title when activeRuntimeId provided', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByText('Model Playground')).toBeDefined()
  })

  it('renders textarea for prompt input', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByPlaceholderText(/Enter a prompt to test/)).toBeDefined()
  })

  it('renders temperature and max tokens sliders', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByTestId('slider-Temperature')).toBeDefined()
    expect(screen.getByTestId('slider-Max tokens')).toBeDefined()
  })

  it('renders Generate button', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    expect(screen.getByText('Generate')).toBeDefined()
  })

  it('disables Generate when prompt is empty', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    const btn = screen.getByText('Generate').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('enables Generate when prompt is entered', () => {
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'hi' } })
    const btn = screen.getByText('Generate').closest('button')!
    expect(btn.disabled).toBe(false)
  })

  it('calls generateStream on Generate click', async () => {
    mockGenerateStream.mockImplementation(async (_req: any, onToken: any, onDone: any) => {
      onToken('hello output')
      onDone()
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'hi' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => {
      expect(mockGenerateStream).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'hi' }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      )
    })
  })

  it('displays generated output', async () => {
    mockGenerateStream.mockImplementation(async (_req: any, onToken: any, onDone: any) => {
      onToken('model response')
      onDone()
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('model response')).toBeDefined())
  })

  it('shows error message on generate failure', async () => {
    mockGenerateStream.mockImplementation(async (_req: any, _onToken: any, _onDone: any, onError: any) => {
      onError('OOM')
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'fail' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('Error: OOM')).toBeDefined())
  })

  it('Clear button resets prompt and output', async () => {
    mockGenerateStream.mockImplementation(async (_req: any, onToken: any, onDone: any) => {
      onToken('done')
      onDone()
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'x' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getByText('done')).toBeDefined())
    fireEvent.click(screen.getByText('Clear'))
    expect((screen.getByPlaceholderText(/Enter a prompt to test/) as HTMLTextAreaElement).value).toBe('')
    expect(screen.queryByText('done')).toBeNull()
  })

  it('shows Generating... while request is pending', async () => {
    let resolveStream: any
    mockGenerateStream.mockImplementation((_req: any, onToken: any, onDone: any) => {
      return new Promise(r => { resolveStream = { onToken, onDone, r } })
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 'wait' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => expect(screen.getAllByText('Generating...').length).toBeGreaterThanOrEqual(1))
    resolveStream.onToken('ok')
    resolveStream.onDone()
    resolveStream.r()
  })

  it('applies temperature to generate call', async () => {
    mockGenerateStream.mockImplementation(async (_req: any, onToken: any, onDone: any) => {
      onToken('')
      onDone()
    })
    render(<ModelPlaygroundCard activeRuntimeId="gpt2" />)
    fireEvent.change(screen.getByPlaceholderText(/Enter a prompt to test/), { target: { value: 't' } })
    fireEvent.click(screen.getByText('Generate'))
    await waitFor(() => {
      expect(mockGenerateStream).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.7, max_new_tokens: 100 }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      )
    })
  })
})
