// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

afterEach(() => { cleanup() })

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <span data-testid="card-title" {...props}>{children}</span>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, className, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} {...props}>{children}</button>
  ),
  Input: ({ value, onChange, onKeyDown, placeholder, ...props }: any) => (
    <input value={value} onChange={onChange} onKeyDown={onKeyDown} placeholder={placeholder} {...props} />
  ),
  IconRefresh: (props: any) => <span />,
  IconPlus: (props: any) => <span />,
  IconTrash: (props: any) => <span />,
}))

vi.mock('@/lib/time-format', () => ({
  formatDateTimeFull: (d: string) => '2025-01-01 12:00:00',
}))

import { WeightSnapshotsManager } from './WeightSnapshotsManager'

const snapshots = [
  { name: 'snap-v1', saved_at: '2025-01-01T12:00:00Z' },
  { name: 'snap-v2', saved_at: '2025-02-01T12:00:00Z' },
]

describe('WeightSnapshotsManager', () => {
  it('renders title', () => {
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getAllByText('Weight Snapshots').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state', () => {
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getAllByText('No snapshots saved yet.').length).toBeGreaterThanOrEqual(1)
  })

  it('renders snapshot list', () => {
    render(<WeightSnapshotsManager snapshots={snapshots} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getAllByText('snap-v1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('snap-v2').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onSave when Save clicked', () => {
    const onSave = vi.fn()
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="new-snap" onSnapshotNameChange={vi.fn()} onSave={onSave} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    const saveButtons = screen.getAllByText('Save')
    fireEvent.click(saveButtons[0])
    expect(onSave).toHaveBeenCalledTimes(1)
  })

  it('disables Save when name is empty', () => {
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    const saveButtons = screen.getAllByText('Save')
    expect(saveButtons[0].closest('button')).toBeDisabled()
  })

  it('calls onLoad when Load clicked', () => {
    const onLoad = vi.fn()
    render(<WeightSnapshotsManager snapshots={snapshots} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={onLoad} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    const loadButtons = screen.getAllByText('Load')
    fireEvent.click(loadButtons[0])
    expect(onLoad).toHaveBeenCalledWith('snap-v1')
  })

  it('calls onDelete when delete clicked', () => {
    const onDelete = vi.fn()
    render(<WeightSnapshotsManager snapshots={snapshots} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={onDelete} onRefresh={vi.fn()} />)
    const deleteButtons = screen.getAllByLabelText('Delete snapshot snap-v1')
    fireEvent.click(deleteButtons[0])
    expect(onDelete).toHaveBeenCalledWith('snap-v1')
  })

  it('calls onRefresh when refresh clicked', () => {
    const onRefresh = vi.fn()
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={onRefresh} />)
    const refreshButton = screen.getByLabelText('Refresh snapshots')
    fireEvent.click(refreshButton)
    expect(onRefresh).toHaveBeenCalledTimes(1)
  })

  it('calls onSnapshotNameChange when typing', () => {
    const onChange = vi.fn()
    render(<WeightSnapshotsManager snapshots={[]} newSnapshotName="" onSnapshotNameChange={onChange} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    const input = screen.getAllByPlaceholderText('Snapshot name...')[0]
    fireEvent.change(input, { target: { value: 'test' } })
    expect(onChange).toHaveBeenCalledWith('test')
  })

  it('shows snapshot dates', () => {
    render(<WeightSnapshotsManager snapshots={snapshots} newSnapshotName="" onSnapshotNameChange={vi.fn()} onSave={vi.fn()} onLoad={vi.fn()} onDelete={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getAllByText('2025-01-01 12:00:00').length).toBeGreaterThanOrEqual(1)
  })
})
