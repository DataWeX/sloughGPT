import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { SmartPromptSuggestions } from './SmartPromptSuggestions'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

describe('SmartPromptSuggestions', () => {
  it('renders title', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    expect(screen.getByText('Quick Prompts')).toBeInTheDocument()
  })

  it('renders built-in prompts', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    expect(screen.getByText('Explain')).toBeInTheDocument()
    expect(screen.getByText('Summarize')).toBeInTheDocument()
  })

  it('calls onSelect when prompt clicked', () => {
    const onSelect = vi.fn()
    render(<SmartPromptSuggestions onSelect={onSelect} />)
    fireEvent.click(screen.getByText('Explain'))
    expect(onSelect).toHaveBeenCalledWith('Explain this concept in simple terms:')
  })

  it('filters by category', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('Code'))
    expect(screen.getByText('Debug')).toBeInTheDocument()
    expect(screen.getByText('Code Review')).toBeInTheDocument()
    expect(screen.queryByText('Explain')).not.toBeInTheDocument()
  })

  it('shows all prompts when Show All clicked', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('Show All'))
    expect(screen.getByText('Brainstorm')).toBeInTheDocument()
  })

  it('opens custom template form', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Custom'))
    expect(screen.getByPlaceholderText('Template name...')).toBeInTheDocument()
  })

  it('creates custom template', async () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('Show All'))
    fireEvent.click(screen.getByText('+ Custom'))
    fireEvent.change(screen.getByPlaceholderText('Template name...'), {
      target: { value: 'My Template' },
    })
    fireEvent.change(screen.getByPlaceholderText('Prompt text...'), {
      target: { value: 'My custom prompt' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    expect(screen.getAllByText('My Template').length).toBeGreaterThan(0)
  })

  it('persists custom templates', async () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Custom'))
    fireEvent.change(screen.getByPlaceholderText('Template name...'), {
      target: { value: 'Saved' },
    })
    fireEvent.change(screen.getByPlaceholderText('Prompt text...'), {
      target: { value: 'Saved prompt' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    const stored = JSON.parse(localStorage.getItem('chat-prompt-templates') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Saved')
  })

  it('shows category filter', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    expect(screen.getByText('Learning')).toBeInTheDocument()
    expect(screen.getByText('Code')).toBeInTheDocument()
  })

  it('selects All category', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('Code'))
    fireEvent.click(screen.getByText('All'))
    expect(screen.getByText('All')).toHaveClass('bg-primary/20')
  })

  it('cancels creation', () => {
    render(<SmartPromptSuggestions onSelect={vi.fn()} />)
    fireEvent.click(screen.getByText('+ Custom'))
    const cancelButtons = screen.getAllByText('Cancel')
    fireEvent.click(cancelButtons[cancelButtons.length - 1])
    expect(screen.queryByPlaceholderText('Template name...')).not.toBeInTheDocument()
  })
})