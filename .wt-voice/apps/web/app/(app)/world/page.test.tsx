import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockStats = vi.fn()
const mockRender = vi.fn()
const mockRenderImage = vi.fn()
const mockNeuralProcess = vi.fn()
const mockTick = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/world-controller', () => ({
  worldController: {
    stats: (...args: unknown[]) => mockStats(...args),
    render: (...args: unknown[]) => mockRender(...args),
    renderImage: (...args: unknown[]) => mockRenderImage(...args),
    neuralProcess: (...args: unknown[]) => mockNeuralProcess(...args),
    tick: (...args: unknown[]) => mockTick(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant }: any) => <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, type, step, className }: any) => (
      <input value={value} onChange={onChange} type={type} step={step} className={className} />
    ),
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
  
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

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

vi.stubGlobal('fetch', vi.fn())

vi.stubGlobal('URL', {
  createObjectURL: vi.fn(() => 'blob:mock-url'),
  revokeObjectURL: vi.fn(),
})

import WorldPage from './page'

describe('WorldPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    render(<WorldPage />)
    expect(screen.getByText('World Render')).toBeInTheDocument()
    expect(screen.getByText('Programmable world simulation and rendering')).toBeInTheDocument()
  })

  it('fetches stats on mount', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: ['physics', 'render'], materials: { grass: 0 } })
    render(<WorldPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('displays components from stats', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: ['physics', 'render'], materials: { grass: 0 } })
    render(<WorldPage />)
    await waitFor(() => {
      expect(screen.getByText('physics')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('render')).toBeInTheDocument()
  })

  it('renders config inputs', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    render(<WorldPage />)
    expect(screen.getByText('Render Config')).toBeInTheDocument()
    expect(screen.getByText('Width')).toBeInTheDocument()
    expect(screen.getByText('Height')).toBeInTheDocument()
    expect(screen.getByText('Samples')).toBeInTheDocument()
  })

  it('renders action buttons', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    render(<WorldPage />)
    expect(screen.getByText('Render')).toBeInTheDocument()
    expect(screen.getByText('Run Tick')).toBeInTheDocument()
    expect(screen.getByText('Tick + Neural')).toBeInTheDocument()
  })

  it('calls render on Render click', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockRender.mockResolvedValue({ shapes: { world: [1, 2] }, tensor_keys: ['world'] })
    mockRenderImage.mockResolvedValue(new Blob(['fake'], { type: 'image/png' }))
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Render'))
    await waitFor(() => {
      expect(mockRender).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockRenderImage).toHaveBeenCalled()
  })

  it('shows success toast on render complete', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockRender.mockResolvedValue({ shapes: {}, tensor_keys: [] })
    mockRenderImage.mockResolvedValue(new Blob(['fake'], { type: 'image/png' }))
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Render'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Render complete', 'success')
    }, { timeout: 5000 })
  })

  it('shows error toast on render failure', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockRender.mockRejectedValue(new Error('OOM'))
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Render'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Render failed: OOM', 'error')
    }, { timeout: 5000 })
  })

  it('calls tick on Run Tick click', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockTick.mockResolvedValue({ tick: 1, babies: 5, render_stats: null })
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Run Tick'))
    await waitFor(() => {
      expect(mockTick).toHaveBeenCalledWith(1, true, false)
    }, { timeout: 5000 })
  })

  it('shows tick result after tick', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockTick.mockResolvedValue({ tick: 3, babies: 5, render_stats: null })
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Run Tick'))
    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('calls neural process on Tick + Neural click', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    mockTick.mockResolvedValue({ tick: 1, babies: 0, render_stats: null })
    mockNeuralProcess.mockResolvedValue({ embedding_shape: [128], descriptor: { foo: 'bar' }, stats: {} })
    render(<WorldPage />)
    fireEvent.click(screen.getByText('Tick + Neural'))
    await waitFor(() => {
      expect(mockTick).toHaveBeenCalledWith(1, true, true)
    }, { timeout: 5000 })
    await waitFor(() => {
      expect(mockNeuralProcess).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('shows materials from stats', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: { grass: 0, water: 1 } })
    render(<WorldPage />)
    await waitFor(() => {
      expect(screen.getByText('grass (0)')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('water (1)')).toBeInTheDocument()
  })

  it('refresh button re-fetches stats', async () => {
    mockStats.mockResolvedValue({ status: 'ok', components: [], materials: {} })
    render(<WorldPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
