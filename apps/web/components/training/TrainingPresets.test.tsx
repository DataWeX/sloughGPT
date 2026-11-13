// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { TrainingPresets } from './TrainingPresets'
import { BUILT_IN_PRESETS } from '@/hooks/useTrainingForm'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, variant, size, className }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
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

vi.mock('@/lib/download-utils', () => ({ downloadJson: vi.fn() }))

afterEach(() => { cleanup() })

describe('TrainingPresets', () => {
  const defaultProps = {
    onApply: vi.fn(),
    customPresets: [],
    onSave: vi.fn(),
    onDelete: vi.fn(),
  }

  beforeEach(() => { vi.clearAllMocks() })

  it('renders preset label', () => {
    render(<TrainingPresets {...defaultProps} />)
    expect(screen.getByText('Presets')).toBeTruthy()
  })

  it('renders all built-in presets as chips', () => {
    render(<TrainingPresets {...defaultProps} />)
    BUILT_IN_PRESETS.forEach(p => {
      expect(screen.getByText(p.name)).toBeTruthy()
    })
  })

  it('calls onApply when a preset chip is clicked', () => {
    const onApply = vi.fn()
    render(<TrainingPresets {...defaultProps} onApply={onApply} />)
    fireEvent.click(screen.getByText('Quick test'))
    expect(onApply).toHaveBeenCalledWith(BUILT_IN_PRESETS[0])
  })

  it('renders custom presets alongside built-in', () => {
    const custom = [{ name: 'My preset', description: 'Custom', method: 'distill' as const, epochs: 5, lr: 0.001, batchSize: 16 }]
    render(<TrainingPresets {...defaultProps} customPresets={custom} />)
    expect(screen.getByText('My preset')).toBeTruthy()
    expect(screen.getByText('Quick test')).toBeTruthy()
  })

  it('shows delete button on custom presets on hover', () => {
    const custom = [{ name: 'My preset', description: 'Custom', method: 'distill' as const, epochs: 5, lr: 0.001, batchSize: 16 }]
    render(<TrainingPresets {...defaultProps} customPresets={custom} />)
    const chip = screen.getByText('My preset').closest('button')!
    expect(chip.querySelector('[role="button"]')).toBeTruthy()
  })

  it('calls onDelete when delete button is clicked', () => {
    const onDelete = vi.fn()
    const custom = [{ name: 'My preset', description: 'Custom', method: 'distill' as const, epochs: 5, lr: 0.001, batchSize: 16 }]
    render(<TrainingPresets {...defaultProps} customPresets={custom} onDelete={onDelete} />)
    const deleteBtn = screen.getByText('My preset').closest('button')!.querySelector('[role="button"]')!
    fireEvent.click(deleteBtn)
    expect(onDelete).toHaveBeenCalledWith('My preset')
  })

  it('shows save form when Save current is clicked', () => {
    render(<TrainingPresets {...defaultProps} />)
    fireEvent.click(screen.getByText('Save current'))
    expect(screen.getByPlaceholderText('Preset name...')).toBeTruthy()
  })

  it('hides save form when Cancel is clicked', () => {
    render(<TrainingPresets {...defaultProps} />)
    fireEvent.click(screen.getByText('Save current'))
    expect(screen.getByPlaceholderText('Preset name...')).toBeTruthy()
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByPlaceholderText('Preset name...')).toBeFalsy()
  })

  it('calls onSave with current form state when getCurrentState is provided', () => {
    const onSave = vi.fn()
    const getCurrentState = vi.fn().mockReturnValue({ name: '', description: '', method: 'finetune', epochs: 10, lr: 0.002, batchSize: 16 })
    render(<TrainingPresets {...defaultProps} onSave={onSave} getCurrentState={getCurrentState} />)
    fireEvent.click(screen.getByText('Save current'))
    const input = screen.getByPlaceholderText('Preset name...')
    fireEvent.change(input, { target: { value: 'My config' } })
    fireEvent.click(screen.getByText('Save'))
    expect(getCurrentState).toHaveBeenCalled()
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ name: 'My config', method: 'finetune', epochs: 10 }))
  })

  it('falls back to defaults when getCurrentState is not provided', () => {
    const onSave = vi.fn()
    render(<TrainingPresets {...defaultProps} onSave={onSave} />)
    fireEvent.click(screen.getByText('Save current'))
    const input = screen.getByPlaceholderText('Preset name...')
    fireEvent.change(input, { target: { value: 'Fallback' } })
    fireEvent.click(screen.getByText('Save'))
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ name: 'Fallback', method: 'distill', epochs: 5 }))
  })

  it('disables Save button when name is empty', () => {
    render(<TrainingPresets {...defaultProps} />)
    fireEvent.click(screen.getByText('Save current'))
    const saveBtn = screen.getByText('Save').closest('button')!
    expect(saveBtn.disabled).toBe(true)
  })

  it('calls onSave on Enter key', () => {
    const onSave = vi.fn()
    render(<TrainingPresets {...defaultProps} onSave={onSave} />)
    fireEvent.click(screen.getByText('Save current'))
    const input = screen.getByPlaceholderText('Preset name...')
    fireEvent.change(input, { target: { value: 'Quick save' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onSave).toHaveBeenCalled()
  })

  it('shows Export button when custom presets exist', () => {
    const custom = [{ name: 'My preset', description: 'Custom', method: 'distill' as const, epochs: 5, lr: 0.001, batchSize: 16 }]
    render(<TrainingPresets {...defaultProps} customPresets={custom} />)
    expect(screen.getByText('Export')).toBeTruthy()
  })

  it('hides Export button when no custom presets', () => {
    render(<TrainingPresets {...defaultProps} />)
    expect(screen.queryByText('Export')).toBeFalsy()
  })

  it('calls downloadJson when Export is clicked', async () => {
    const { downloadJson } = await import('@/lib/download-utils')
    const custom = [{ name: 'My preset', description: 'Custom', method: 'distill' as const, epochs: 5, lr: 0.001, batchSize: 16 }]
    render(<TrainingPresets {...defaultProps} customPresets={custom} />)
    fireEvent.click(screen.getByText('Export'))
    expect(downloadJson).toHaveBeenCalledWith(custom, 'training-presets.json')
  })

  it('shows Import button', () => {
    render(<TrainingPresets {...defaultProps} />)
    expect(screen.getByText('Import')).toBeTruthy()
  })
})
