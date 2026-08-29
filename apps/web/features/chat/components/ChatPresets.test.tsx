import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatPresets } from './ChatPresets'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

describe('ChatPresets', () => {
  it('renders empty state', () => {
    render(<ChatPresets onSelect={vi.fn()} />)
    expect(screen.getByText('No presets yet. Click + to create one.')).toBeInTheDocument()
  })

  it('opens create form when plus clicked', () => {
    render(<ChatPresets onSelect={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByPlaceholderText('Preset name...')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Prompt template...')).toBeInTheDocument()
  })

  it('creates a preset', async () => {
    const onSelect = vi.fn()
    render(<ChatPresets onSelect={onSelect} />)
    
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Preset name...'), { target: { value: 'My Preset' } })
    fireEvent.change(screen.getByPlaceholderText('Prompt template...'), { target: { value: 'Hello world' } })
    
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    
    expect(screen.getByText('My Preset')).toBeInTheDocument()
    expect(screen.getByText(/Hello world/)).toBeInTheDocument()
  })

  it('calls onSelect when preset clicked', async () => {
    const onSelect = vi.fn()
    render(<ChatPresets onSelect={onSelect} />)
    
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Preset name...'), { target: { value: 'Test' } })
    fireEvent.change(screen.getByPlaceholderText('Prompt template...'), { target: { value: 'Prompt text' } })
    
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    
    fireEvent.click(screen.getByText('Test'))
    expect(onSelect).toHaveBeenCalledWith('Prompt text')
  })

  it('persists to localStorage', async () => {
    render(<ChatPresets onSelect={vi.fn()} />)
    
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Preset name...'), { target: { value: 'Saved' } })
    fireEvent.change(screen.getByPlaceholderText('Prompt template...'), { target: { value: 'Content' } })
    
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    
    const stored = JSON.parse(localStorage.getItem('chat-presets') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Saved')
  })

  it('deletes a preset', async () => {
    render(<ChatPresets onSelect={vi.fn()} />)
    
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Preset name...'), { target: { value: 'Delete me' } })
    fireEvent.change(screen.getByPlaceholderText('Prompt template...'), { target: { value: 'x' } })
    
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    
    fireEvent.click(screen.getByTitle('Delete preset'))
    expect(screen.getByText('No presets yet. Click + to create one.')).toBeInTheDocument()
  })

  it('disables save when fields empty', () => {
    render(<ChatPresets onSelect={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    const saveBtn = screen.getByText('Save').closest('button')!
    expect(saveBtn).toBeDisabled()
  })
})