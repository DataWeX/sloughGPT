import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ContextWindowViewer, type ContextWindowItem } from './ContextWindowViewer'

afterEach(cleanup)

const mockItems: ContextWindowItem[] = [
  { label: 'System instructions', content: 'You are helpful.', tokenCount: 10, type: 'system' },
  { label: 'Knowledge base', content: 'Retrieved context here.', tokenCount: 20, type: 'knowledge' },
  { label: 'User message', content: 'Hello world', tokenCount: 3, type: 'user' },
]

describe('ContextWindowViewer', () => {
  it('renders empty state when no items', () => {
    render(<ContextWindowViewer items={[]} />)
    expect(screen.getByText('No context items loaded')).toBeInTheDocument()
  })

  it('renders item count and token estimate', () => {
    render(<ContextWindowViewer items={mockItems} />)
    expect(screen.getByText('3 items · ~33 tokens')).toBeInTheDocument()
  })

  it('renders item labels', () => {
    render(<ContextWindowViewer items={mockItems} />)
    expect(screen.getByText('System instructions')).toBeInTheDocument()
    expect(screen.getByText('Knowledge base')).toBeInTheDocument()
    expect(screen.getByText('User message')).toBeInTheDocument()
  })

  it('expands item on click', () => {
    render(<ContextWindowViewer items={mockItems} />)
    fireEvent.click(screen.getByText('System instructions'))
    expect(screen.getByText('You are helpful.')).toBeInTheDocument()
  })

  it('collapses expanded item on second click', () => {
    render(<ContextWindowViewer items={mockItems} />)
    fireEvent.click(screen.getByText('System instructions'))
    expect(screen.getByText('You are helpful.')).toBeInTheDocument()
    fireEvent.click(screen.getByText('System instructions'))
    expect(screen.queryByText('You are helpful.')).not.toBeInTheDocument()
  })

  it('expand all button works', () => {
    render(<ContextWindowViewer items={mockItems} />)
    fireEvent.click(screen.getByText('Expand All'))
    expect(screen.getByText('You are helpful.')).toBeInTheDocument()
    expect(screen.getByText('Retrieved context here.')).toBeInTheDocument()
    expect(screen.getByText('Hello world')).toBeInTheDocument()
  })

  it('collapse all button works', () => {
    render(<ContextWindowViewer items={mockItems} />)
    fireEvent.click(screen.getByText('Expand All'))
    fireEvent.click(screen.getByText('Collapse All'))
    expect(screen.queryByText('You are helpful.')).not.toBeInTheDocument()
  })

  it('shows type badges', () => {
    render(<ContextWindowViewer items={mockItems} />)
    expect(screen.getByText('System Prompt')).toBeInTheDocument()
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
  })

  it('shows token counts per item', () => {
    render(<ContextWindowViewer items={mockItems} />)
    expect(screen.getByText('~10 tokens')).toBeInTheDocument()
    expect(screen.getByText('~20 tokens')).toBeInTheDocument()
  })

  it('uses provided totalTokens when available', () => {
    render(<ContextWindowViewer items={mockItems} totalTokens={100} />)
    expect(screen.getByText('3 items · ~100 tokens')).toBeInTheDocument()
  })
})