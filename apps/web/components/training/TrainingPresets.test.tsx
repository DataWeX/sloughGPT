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
}))

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
})
