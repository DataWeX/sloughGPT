import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  IconX: (p: any) => <svg {...p} />,
  IconPlus: (p: any) => <svg {...p} />,
  IconTrash: (p: any) => <svg {...p} />,
  IconCheck: (p: any) => <svg {...p} />,
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

import { TemplateDialog } from './TemplateDialog'

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  onSelect: vi.fn(),
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
})

afterEach(cleanup)

describe('TemplateDialog', () => {
  it('renders nothing when closed', () => {
    render(<TemplateDialog {...defaultProps} open={false} />)
    expect(screen.queryByText('Conversation Templates')).toBeNull()
  })

  it('renders when open', () => {
    render(<TemplateDialog {...defaultProps} />)
    expect(screen.getByText('Conversation Templates')).toBeDefined()
    expect(screen.getByPlaceholderText('Template name')).toBeDefined()
    expect(screen.getByPlaceholderText(/Template content/)).toBeDefined()
  })

  it('shows empty state when no templates', () => {
    render(<TemplateDialog {...defaultProps} />)
    expect(screen.getByText('No templates yet. Create one above.')).toBeDefined()
  })

  it('creates a new template', () => {
    render(<TemplateDialog {...defaultProps} />)
    const nameInput = screen.getByPlaceholderText('Template name')
    const contentArea = screen.getByPlaceholderText(/Template content/)

    fireEvent.change(nameInput, { target: { value: 'My Template' } })
    fireEvent.change(contentArea, { target: { value: 'Hello world' } })
    fireEvent.click(screen.getByText('Save'))

    expect(screen.getByText('My Template')).toBeDefined()
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('disables save when name or content empty', () => {
    render(<TemplateDialog {...defaultProps} />)
    const saveBtn = screen.getByText('Save')
    expect(saveBtn.closest('button')?.disabled).toBe(true)
  })

  it('saves to localStorage', () => {
    render(<TemplateDialog {...defaultProps} />)
    fireEvent.change(screen.getByPlaceholderText('Template name'), { target: { value: 'Test' } })
    fireEvent.change(screen.getByPlaceholderText(/Template content/), { target: { value: 'Content' } })
    fireEvent.click(screen.getByText('Save'))

    const stored = JSON.parse(localStorage.getItem('chat-templates') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Test')
  })

  it('loads templates from localStorage', () => {
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Saved Template', content: 'Saved content', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} />)
    expect(screen.getByText('Saved Template')).toBeDefined()
    expect(screen.getByText('Saved content')).toBeDefined()
    expect(screen.getByText('1 templates saved')).toBeDefined()
  })

  it('deletes a template', () => {
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'To Delete', content: 'Delete me', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} />)
    fireEvent.click(screen.getByTitle('Delete template'))
    expect(screen.getByText('No templates yet. Create one above.')).toBeDefined()
    expect(JSON.parse(localStorage.getItem('chat-templates') || '[]')).toHaveLength(0)
  })

  it('enters edit mode for a template', () => {
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Edit Me', content: 'Original content', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} />)
    fireEvent.click(screen.getByTitle('Edit template'))

    expect(screen.getByDisplayValue('Edit Me')).toBeDefined()
    expect(screen.getByDisplayValue('Original content')).toBeDefined()
    expect(screen.getByText('Editing template')).toBeDefined()
    expect(screen.getByText('Update')).toBeDefined()
    expect(screen.getByText('Cancel')).toBeDefined()
  })

  it('cancels edit mode', () => {
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Edit Me', content: 'Original', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} />)
    fireEvent.click(screen.getByTitle('Edit template'))
    fireEvent.click(screen.getByText('Cancel'))

    expect(screen.queryByText('Editing template')).toBeNull()
    expect(screen.getByPlaceholderText('Template name')).toHaveValue('')
  })

  it('updates an existing template', () => {
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Old Name', content: 'Old Content', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} />)
    fireEvent.click(screen.getByTitle('Edit template'))
    fireEvent.change(screen.getByDisplayValue('Old Name'), { target: { value: 'New Name' } })
    fireEvent.click(screen.getByText('Update'))

    expect(screen.getByText('New Name')).toBeDefined()
    const stored = JSON.parse(localStorage.getItem('chat-templates') || '[]')
    expect(stored[0].name).toBe('New Name')
  })

  it('calls onSelect and onClose when using a template', () => {
    const onSelect = vi.fn()
    const onClose = vi.fn()
    localStorage.setItem('chat-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Use Me', content: 'Use this', createdAt: Date.now() },
    ]))
    render(<TemplateDialog {...defaultProps} onSelect={onSelect} onClose={onClose} />)

    const useBtn = screen.getByTitle('Use template')
    fireEvent.click(useBtn)

    expect(onSelect).toHaveBeenCalledWith('Use this')
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on backdrop click', () => {
    const onClose = vi.fn()
    render(<TemplateDialog {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getByText('Conversation Templates').closest('[class*="bg-black"]')!)
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on X button click', () => {
    const onClose = vi.fn()
    render(<TemplateDialog {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText('Close'))
    expect(onClose).toHaveBeenCalled()
  })
})
