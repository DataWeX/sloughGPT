import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatExportTemplates } from './ChatExportTemplates'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
    writable: true,
  })
})

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: new Date(Date.now() - 60000) },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: new Date() },
]

describe('ChatExportTemplates', () => {
  it('renders template list', () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    expect(screen.getByText('Full Export')).toBeInTheDocument()
    expect(screen.getByText('Conversation Only')).toBeInTheDocument()
    expect(screen.getByText('Code Snippets')).toBeInTheDocument()
    expect(screen.getByText('Structured Data')).toBeInTheDocument()
  })

  it('selects a template', () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByText('Conversation Only'))
    expect(screen.getByText('Conversation Only').closest('[class*="bg-primary"]')).toBeInTheDocument()
  })

  it('shows export buttons', () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    expect(screen.getByText('Download')).toBeInTheDocument()
    expect(screen.getByText('Copy')).toBeInTheDocument()
  })

  it('disables buttons when no messages', () => {
    render(<ChatExportTemplates messages={[]} />)
    expect(screen.getByText('Download')).toBeDisabled()
    expect(screen.getByText('Copy')).toBeDisabled()
  })

  it('opens custom template form', () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByLabelText('Create template'))
    expect(screen.getByPlaceholderText('Template name...')).toBeInTheDocument()
  })

  it('creates custom template', async () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByLabelText('Create template'))
    fireEvent.change(screen.getByPlaceholderText('Template name...'), { target: { value: 'My Template' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    expect(screen.getByText('My Template')).toBeInTheDocument()
  })

  it('creates template on Enter', async () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByLabelText('Create template'))
    const input = screen.getByPlaceholderText('Template name...')
    fireEvent.change(input, { target: { value: 'EnterTest' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getByText('EnterTest')).toBeInTheDocument()
  })

  it('deletes custom template', async () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByLabelText('Create template'))
    fireEvent.change(screen.getByPlaceholderText('Template name...'), { target: { value: 'Delete Me' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    fireEvent.click(screen.getByTitle('Delete template'))
    expect(screen.queryByText('Delete Me')).not.toBeInTheDocument()
  })

  it('persists templates to localStorage', async () => {
    render(<ChatExportTemplates messages={mockMessages} />)
    fireEvent.click(screen.getByLabelText('Create template'))
    fireEvent.change(screen.getByPlaceholderText('Template name...'), { target: { value: 'Saved' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    const stored = JSON.parse(localStorage.getItem('chat-export-templates') || '[]')
    expect(stored.length).toBeGreaterThanOrEqual(5)
  })
})