import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const quickPromptsModule = vi.hoisted(() => {
  const data = {
    writing: [
      { id: 'p1', name: 'Blog Post', description: 'Write a blog', prompt: 'Write a blog about {{text}}', icon: '✍️', category: 'writing' },
    ],
    custom: [
      { id: 'p2', name: 'My Prompt', description: 'Custom', prompt: 'Custom prompt', icon: '⚡', category: 'custom' },
    ],
  }
  return {
    listPromptsByCategory: vi.fn(() => data),
    createPrompt: vi.fn(),
    updatePrompt: vi.fn(),
    deletePrompt: vi.fn(),
    resetToDefaults: vi.fn(),
    applyPrompt: vi.fn((p: any) => p.prompt),
  }
})

vi.mock('@/lib/quick-prompts', () => quickPromptsModule)

import { QuickPrompts } from './QuickPrompts'

describe('QuickPrompts', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders category labels', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    expect(screen.getByText('Writing')).toBeDefined()
    const customLabels = screen.getAllByText('Custom')
    expect(customLabels.length).toBeGreaterThanOrEqual(1)
  })

  it('renders prompt names', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    expect(screen.getByText('Blog Post')).toBeDefined()
    expect(screen.getByText('My Prompt')).toBeDefined()
  })

  it('filters prompts by search', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search prompts...'), { target: { value: 'Blog' } })
    expect(screen.getByText('Blog Post')).toBeDefined()
    expect(screen.queryByText('My Prompt')).toBeNull()
  })

  it('shows empty state when no prompts match search', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search prompts...'), { target: { value: 'zzzzz' } })
    expect(screen.getByText('No prompts found')).toBeDefined()
  })

  it('calls onUsePrompt when a prompt is clicked', () => {
    const onUse = vi.fn()
    render(<QuickPrompts onUsePrompt={onUse} />)
    fireEvent.click(screen.getByText('Blog Post'))
    expect(onUse).toHaveBeenCalledWith('Write a blog about {{text}}')
  })

  it('shows create form when + New clicked', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    fireEvent.click(screen.getByText('+ New'))
    expect(screen.getByPlaceholderText('Name')).toBeDefined()
  })

  it('calls createPrompt when save in create form', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'New Prompt' } })
    fireEvent.change(screen.getByPlaceholderText('Prompt template. Use {{text}} where user input goes.'), { target: { value: 'Template' } })
    fireEvent.click(screen.getByText('Save'))
    expect(quickPromptsModule.createPrompt).toHaveBeenCalled()
  })

  it('shows Edit and Delete buttons on hover for custom prompts', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    expect(screen.getAllByText('Edit').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Delete').length).toBeGreaterThanOrEqual(1)
  })

  it('shows custom badge for custom prompts', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    const badges = screen.getAllByText('custom')
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })

  it('calls deletePrompt when Delete clicked', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    const deleteBtns = screen.getAllByText('Delete')
    fireEvent.click(deleteBtns[0])
    expect(quickPromptsModule.deletePrompt).toHaveBeenCalledWith('p2')
  })

  it('calls resetToDefaults when Reset clicked', () => {
    render(<QuickPrompts onUsePrompt={vi.fn()} />)
    fireEvent.click(screen.getByText('Reset'))
    expect(quickPromptsModule.resetToDefaults).toHaveBeenCalled()
  })
})
