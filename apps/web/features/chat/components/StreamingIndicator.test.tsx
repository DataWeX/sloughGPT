import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { StreamingIndicator } from './StreamingIndicator'

afterEach(cleanup)

describe('StreamingIndicator', () => {
  it('renders thinking status', () => {
    render(<StreamingIndicator status="thinking" />)
    expect(screen.getByText('Thinking...')).toBeInTheDocument()
  })

  it('renders generating status', () => {
    render(<StreamingIndicator status="generating" />)
    expect(screen.getByText('Generating...')).toBeInTheDocument()
  })

  it('renders tool call status with tool name', () => {
    render(<StreamingIndicator status="tool_call" toolName="calculator" />)
    expect(screen.getByText('Running tool...')).toBeInTheDocument()
    expect(screen.getByText('(calculator)')).toBeInTheDocument()
  })

  it('renders context status', () => {
    render(<StreamingIndicator status="context" />)
    expect(screen.getByText('Processing context...')).toBeInTheDocument()
  })

  it('renders error status', () => {
    render(<StreamingIndicator status="error" />)
    expect(screen.getByText('Error occurred')).toBeInTheDocument()
  })

  it('has correct aria attributes', () => {
    render(<StreamingIndicator status="thinking" />)
    const indicator = screen.getByRole('status')
    expect(indicator).toHaveAttribute('aria-live', 'polite')
  })
})
