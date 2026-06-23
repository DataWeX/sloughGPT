// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockCtx = {
  model: {
    availableModels: ['gpt2', 'gpt2-medium', 'gpt2-large'],
    current: 'gpt2',
    loading: null,
    generating: false,
    infoMap: {} as Record<string, { cached?: boolean; size_gb?: number }>,
    descriptions: {} as Record<string, string>,
    downloadProgress: {},
    onSelect: vi.fn(),
    onUnload: vi.fn(),
  },
}
vi.mock('@/contexts/ChatToolbarContext', () => ({ useChatToolbarContext: () => mockCtx }))

vi.mock('@/components/ui/dropdown-menu', () => {
  function DM({ children }: any) { return <div>{children}</div> }
  function DMT({ children, asChild, ...props }: any) { return asChild ? <>{children}</> : <button>{children}</button> }
  function DMI({ children, onSelect, disabled }: any) {
    return <button role="menuitem" disabled={disabled} onClick={onSelect}>{children}</button>
  }
  return {
    DropdownMenu: DM, DropdownMenuTrigger: DMT, DropdownMenuContent: ({ children }: any) => <div>{children}</div>,
    DropdownMenuItem: DMI, DropdownMenuSeparator: () => <hr />, DropdownMenuLabel: ({ children }: any) => <div>{children}</div>,
    DropdownMenuCheckboxItem: DMI, DropdownMenuPortal: ({ children }: any) => <div>{children}</div>,
    DropdownMenuGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSub: ({ children }: any) => <div>{children}</div>, DropdownMenuRadioGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSubTrigger: DMT, DropdownMenuSubContent: ({ children }: any) => <div>{children}</div>,
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

  it('shows Unload option when model loaded and onUnload provided', () => {
    render(<ModelDropdown />)
    fireEvent.click(screen.getByLabelText(/Current:/))
    expect(screen.getByText('Remove model')).toBeDefined()
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
})
