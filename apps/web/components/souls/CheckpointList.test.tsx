// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

afterEach(() => {
  cleanup()
})

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => (
    <div data-testid="card" {...props}>
      {children}
    </div>
  ),
  CardHeader: ({ children, ...props }: any) => (
    <div data-testid="card-header" {...props}>
      {children}
    </div>
  ),
  CardTitle: ({ children, ...props }: any) => (
    <span data-testid="card-title" {...props}>
      {children}
    </span>
  ),
  CardContent: ({ children, ...props }: any) => (
    <div data-testid="card-content" {...props}>
      {children}
    </div>
  ),
  Button: ({ children, onClick, disabled, className, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} {...props}>
      {children}
    </button>
  ),
  SearchInput: ({ value, onChange, placeholder, ...props }: any) => (
    <input
      value={value}
      onChange={(e: any) => onChange(e.target.value)}
      placeholder={placeholder}
      {...props}
    />
  ),
  Spinner: (props: any) => <span data-testid="spinner" />,
  IconRefresh: (props: any) => <span />,
  IconDownload: (props: any) => <span />,
  IconTrash: (props: any) => <span />,
}))

vi.mock('@/vite/next-compat/link', () => ({
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}))

vi.mock('@/lib/time-format', () => ({
  formatShortDate: (d: string) => 'Jan 1, 2025',
}))

import { CheckpointList } from './CheckpointList'
import type { Checkpoint } from '@/lib/souls-controller'

const makeCp = (overrides: Partial<Checkpoint> = {}): Checkpoint => ({
  name: 'cp-001',
  soul: 'test-soul',
  loss: 0.5,
  ...overrides,
})

const checkpoints: Checkpoint[] = [
  makeCp({ name: 'cp-001', soul: 'alpha', verdict: 'improved', is_loaded: true }),
  makeCp({ name: 'cp-002', soul: 'beta', verdict: 'degraded' }),
  makeCp({ name: 'cp-003', soul: 'gamma' }),
]

describe('CheckpointList', () => {
  it('renders title with count', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText(/Checkpoints/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders checkpoint items', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText('cp-001').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('cp-002').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('cp-003').length).toBeGreaterThanOrEqual(1)
  })

  it('shows verdict badges', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText('Improved').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Degraded').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loaded badge', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText('loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLoad when Load clicked', () => {
    const onLoad = vi.fn()
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={onLoad}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    const loadButtons = screen.getAllByText('Load')
    fireEvent.click(loadButtons[0])
    expect(onLoad).toHaveBeenCalledWith('cp-001')
  })

  it('shows spinner when loading', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint="cp-001"
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByTestId('spinner').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onDelete when delete clicked', () => {
    const onDelete = vi.fn()
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={onDelete}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    const deleteButtons = screen.getAllByLabelText('Delete checkpoint')
    fireEvent.click(deleteButtons[0])
    expect(onDelete).toHaveBeenCalledWith('cp-001')
  })

  it('calls onInfo when checkpoint name clicked', () => {
    const onInfo = vi.fn()
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={onInfo}
        onRefresh={vi.fn()}
      />,
    )
    const nameEl = screen.getAllByText('cp-001')[0]
    fireEvent.click(nameEl)
    expect(onInfo).toHaveBeenCalledWith('cp-001')
  })

  it('shows empty state when no checkpoints', () => {
    render(
      <CheckpointList
        checkpoints={[]}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText('No checkpoints found.').length).toBeGreaterThanOrEqual(1)
  })

  it('filters by search query', () => {
    render(
      <CheckpointList
        checkpoints={checkpoints}
        searchQuery="cp-001"
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getAllByText('cp-001').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('cp-002')).toBeNull()
  })

  it('shows declared provenance as a badge', () => {
    render(
      <CheckpointList
        checkpoints={[
          makeCp({ name: 'cp-declared', soul: 'alpha', provenance: 'training' }),
          makeCp({ name: 'cp-unknown', soul: 'beta' }),
        ]}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    // Provenance is surfaced when declared...
    expect(screen.getAllByText('training').length).toBe(1)
    // ...and absent when not — an unknown provenance is never invented from
    // the filename, so no badge renders for it.
    expect(screen.getByText('beta').closest('div')?.textContent).not.toContain('export')
  })

  it('renders the integrity hash truncated, keeping the full value on hover', () => {
    render(
      <CheckpointList
        checkpoints={[makeCp({ integrity_hash: 'd9676fd9ddc9cb22' })]}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    // The list summarises; the full hash stays one hover away.
    expect(screen.getByText(/· id d9676fd9$/)).toBeTruthy()
    expect(screen.getByTitle(/d9676fd9ddc9cb22/)).toBeTruthy()
  })

  it('omits identity fields when the sidecar carries none', () => {
    render(
      <CheckpointList
        checkpoints={[makeCp({ name: 'cp-bare' })]}
        searchQuery=""
        onSearchChange={vi.fn()}
        loadingCheckpoint={null}
        onLoad={vi.fn()}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
        onInfo={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    // Checkpoints written before these fields existed must not gain a
    // placeholder row — an empty axis stays invisible rather than shown blank.
    expect(screen.queryByText(/· id /)).toBeNull()
    expect(screen.queryByText('training')).toBeNull()
  })
})
