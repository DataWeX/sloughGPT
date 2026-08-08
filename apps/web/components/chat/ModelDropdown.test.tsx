import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockCtx: {
  model: {
    availableModels: string[];
    current: string | null;
    loading: string | null;
    generating: boolean;
    infoMap: Record<string, { cached?: boolean; size_gb?: number }>;
    descriptions: Record<string, string>;
    downloadProgress: Record<string, any>;
    onSelect: ReturnType<typeof vi.fn>;
    onUnload: ReturnType<typeof vi.fn>;
    fineTuned?: {
      models: Array<{ name: string; model?: string; dataset?: string; size_mb?: number }>;
      loading: boolean;
      onLoad: ReturnType<typeof vi.fn>;
    };
  }
} = {
  model: {
    availableModels: ['gpt2', 'gpt2-medium', 'gpt2-large'],
    current: 'gpt2',
    loading: null,
    generating: false,
    infoMap: {},
    descriptions: {},
    downloadProgress: {},
    onSelect: vi.fn(),
    onUnload: vi.fn(),
    fineTuned: { models: [], loading: false, onLoad: vi.fn() },
  },
}
vi.mock('@/contexts/ChatToolbarContext', () => ({ useChatToolbarContext: () => mockCtx }))

vi.mock('@sloughgpt/strui', () => {
  function DM({ children }: any) { return <div>{children}</div> }
  function DMT({ children, asChild, ...props }: any) { return asChild ? <>{children}</> : <button>{children}</button> }
  function DMI({ children, onSelect, disabled }: any) {
    return <button role="menuitem" disabled={disabled} onClick={onSelect}>{children}</button>
  }
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    DropdownMenu: DM, DropdownMenuTrigger: DMT, DropdownMenuContent: ({ children }: any) => <div>{children}</div>,
    DropdownMenuItem: DMI, DropdownMenuSeparator: () => <hr />, DropdownMenuLabel: ({ children }: any) => <div>{children}</div>,
    DropdownMenuCheckboxItem: DMI, DropdownMenuPortal: ({ children }: any) => <div>{children}</div>,
    DropdownMenuGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSub: ({ children }: any) => <div>{children}</div>, DropdownMenuRadioGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSubTrigger: DMT, DropdownMenuSubContent: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
    IconChevronDown: () => <span data-testid="icon-chevron-down">▼</span>,
    IconCheck: () => <span data-testid="icon-check">✓</span>,
    IconRefresh: () => <span data-testid="icon-refresh">↻</span>,
  }
})

import { ModelDropdown } from './ModelDropdown'

describe('ModelDropdown', () => {
  afterEach(cleanup)

  it('renders trigger with current model name', () => {
    render(<ModelDropdown />)
    const matches = screen.getAllByText('gpt2')
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('renders trigger with "Select model" when no current', () => {
    mockCtx.model.current = null
    render(<ModelDropdown />)
    expect(screen.getByText('Select model')).toBeDefined()
    mockCtx.model.current = 'gpt2'
  })

  it('has aria-label describing current model', () => {
    render(<ModelDropdown />)
    expect(screen.getByLabelText(/Current: gpt2/)).toBeDefined()
  })

  it('renders panel variant with No models available when empty', () => {
    mockCtx.model.availableModels = []
    render(<ModelDropdown variant="panel" />)
    expect(screen.getByText('No models available')).toBeDefined()
    mockCtx.model.availableModels = ['gpt2', 'gpt2-medium', 'gpt2-large']
  })

  it('renders panel variant with models list', () => {
    render(<ModelDropdown variant="panel" panelTitle="Backend Model" />)
    expect(screen.getByText('Backend Model')).toBeDefined()
    expect(screen.getByText('gpt2')).toBeDefined()
    expect(screen.getByText('gpt2-medium')).toBeDefined()
  })

  it('shows download progress in dropdown for loading model', () => {
    mockCtx.model.loading = 'gpt2'
    mockCtx.model.downloadProgress = { gpt2: { percentage: 45, status: 'downloading', speed_mb_per_sec: 10, eta_seconds: 30 } }
    render(<ModelDropdown />)
    const matches = screen.getAllByText('45%')
    expect(matches.length).toBeGreaterThanOrEqual(1)
    mockCtx.model.loading = null
    mockCtx.model.downloadProgress = {}
  })

  it('calls onSelect when model clicked in dropdown', () => {
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    const items = screen.getAllByText('gpt2-large')
    fireEvent.click(items[0])
    expect(mockCtx.model.onSelect).toHaveBeenCalledWith('gpt2-large')
  })

  it('shows generating indicator when generating', () => {
    mockCtx.model.generating = true
    render(<ModelDropdown />)
    const ping = document.querySelector('.animate-ping')
    expect(ping).toBeDefined()
    mockCtx.model.generating = false
  })

  it('download progress bar uses actual percentage not hardcoded', () => {
    mockCtx.model.loading = 'gpt2'
    mockCtx.model.downloadProgress = { gpt2: { percentage: 73, status: 'downloading', speed_mb_per_sec: 15, eta_seconds: 10 } }
    render(<ModelDropdown />)
    const bar = document.querySelector('[style*="width: 73%"]')
    expect(bar).not.toBeNull()
    mockCtx.model.loading = null
    mockCtx.model.downloadProgress = {}
  })

  it('download progress bar falls back to 100% when no percentage', () => {
    mockCtx.model.loading = 'gpt2'
    mockCtx.model.downloadProgress = { gpt2: { status: 'loading', percentage: 0 } }
    render(<ModelDropdown />)
    const bar = document.querySelector('[style*="width: 100%"]')
    expect(bar).not.toBeNull()
    mockCtx.model.loading = null
    mockCtx.model.downloadProgress = {}
  })

  it('strips org prefix from model names in dropdown items', () => {
    mockCtx.model.availableModels = ['Qwen/Qwen2.5-0.5B-Instruct', 'gpt2']
    render(<ModelDropdown variant="panel" />)
    expect(screen.getByText('Qwen2.5-0.5B-Instruct')).toBeDefined()
    expect(screen.getByText('gpt2')).toBeDefined()
    mockCtx.model.availableModels = ['gpt2', 'gpt2-medium', 'gpt2-large']
  })

  it('shows warning dot when model is loading', () => {
    mockCtx.model.loading = 'gpt2'
    render(<ModelDropdown />)
    const dot = document.querySelector('.animate-pulse')
    expect(dot).not.toBeNull()
    mockCtx.model.loading = null
  })

  it('shows success dot when model is loaded', () => {
    render(<ModelDropdown />)
    const dots = document.querySelectorAll('.bg-success')
    expect(dots.length).toBeGreaterThanOrEqual(1)
  })

  it('shows muted dot when no model is loaded', () => {
    mockCtx.model.current = null
    render(<ModelDropdown />)
    const dots = document.querySelectorAll('.bg-muted-foreground\\/30')
    expect(dots.length).toBeGreaterThanOrEqual(1)
    mockCtx.model.current = 'gpt2'
  })

  it('hides fine-tuned section when no fine-tuned models exist', () => {
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    expect(screen.queryByText('Fine-tuned')).toBeNull()
  })

  it('renders fine-tuned models with name and size', () => {
    mockCtx.model.fineTuned = {
      models: [{ name: 'gpt2__dataset_1', model: 'gpt2', dataset: 'dataset_1', size_mb: 14.2 }],
      loading: false,
      onLoad: vi.fn(),
    }
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    expect(screen.getByText('Fine-tuned')).toBeDefined()
    const matches = screen.getAllByText('gpt2 · dataset_1')
    expect(matches.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('14.2 MB')).toBeDefined()
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
  })

  it('calls onLoad when fine-tuned model selected', () => {
    const onLoad = vi.fn()
    mockCtx.model.fineTuned = {
      models: [{ name: 'gpt2__dataset_1' }],
      loading: false,
      onLoad,
    }
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    const item = screen.getAllByText('gpt2 · dataset_1')
    fireEvent.click(item[0])
    expect(onLoad).toHaveBeenCalledWith('gpt2__dataset_1')
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
  })

  it('disables fine-tuned item while loading and shows check when loaded', () => {
    mockCtx.model.fineTuned = {
      models: [{ name: 'gpt2__dataset_1' }],
      loading: true,
      onLoad: vi.fn(),
    }
    mockCtx.model.loading = 'gpt2__dataset_1'
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    const item = screen.getByRole('menuitem', { name: /gpt2 · dataset_1/ })
    expect(item).toBeDisabled()
    mockCtx.model.loading = null
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
  })

  it('does not mark fine-tuned item loaded when only its base model is current', () => {
    mockCtx.model.fineTuned = {
      models: [{ name: 'gpt2__dataset_1', model: 'gpt2', dataset: 'dataset_1' }],
      loading: false,
      onLoad: vi.fn(),
    }
    mockCtx.model.current = 'gpt2'
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    const item = screen.getByRole('menuitem', { name: /gpt2 · dataset_1/ })
    expect(item.querySelector('svg')).toBeNull()
    mockCtx.model.current = null
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
  })

  it('marks fine-tuned item loaded when its dir name is current', () => {
    mockCtx.model.fineTuned = {
      models: [{ name: 'gpt2__dataset_1', model: 'gpt2', dataset: 'dataset_1' }],
      loading: false,
      onLoad: vi.fn(),
    }
    mockCtx.model.current = 'gpt2__dataset_1'
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    const item = screen.getByRole('menuitem', { name: /gpt2 · dataset_1/ })
    expect(item.querySelector('svg')).toBeDefined()
    mockCtx.model.current = null
    mockCtx.model.fineTuned = { models: [], loading: false, onLoad: vi.fn() }
  })
})
