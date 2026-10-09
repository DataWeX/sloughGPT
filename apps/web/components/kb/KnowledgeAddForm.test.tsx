/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { KnowledgeAddForm } from './KnowledgeAddForm'

afterEach(() => cleanup())

const defaultProps = {
  content: '',
  topic: 'general',
  importance: 0.7,
  loading: false,
  suggestResult: null,
  onContentChange: vi.fn(),
  onTopicChange: vi.fn(),
  onImportanceChange: vi.fn(),
  onAdd: vi.fn(),
  onSuggest: vi.fn(),
}

describe('KnowledgeAddForm', () => {
  it('renders title', () => {
    render(<KnowledgeAddForm {...defaultProps} />)
    expect(screen.getByText('Add Knowledge Entry')).toBeInTheDocument()
  })

  it('renders content textarea', () => {
    render(<KnowledgeAddForm {...defaultProps} />)
    expect(screen.getByPlaceholderText('Enter knowledge content...')).toBeInTheDocument()
  })

  it('renders Add Entry button', () => {
    render(<KnowledgeAddForm {...defaultProps} />)
    expect(screen.getByText('Add Entry')).toBeInTheDocument()
  })

  it('renders Suggest Topic button', () => {
    render(<KnowledgeAddForm {...defaultProps} />)
    expect(screen.getByText('Suggest Topic')).toBeInTheDocument()
  })

  it('shows suggested topic when provided', () => {
    render(<KnowledgeAddForm {...defaultProps} suggestResult="science" />)
    expect(screen.getByText('Suggested: science')).toBeInTheDocument()
  })

  it('disables Add Entry when content is empty', () => {
    render(<KnowledgeAddForm {...defaultProps} />)
    expect(screen.getByText('Add Entry')).toBeDisabled()
  })

  it('enables Add Entry when content is provided', () => {
    render(<KnowledgeAddForm {...defaultProps} content="some content" />)
    expect(screen.getByText('Add Entry')).not.toBeDisabled()
  })
})
