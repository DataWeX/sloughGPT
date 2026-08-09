import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { ReasoningPanel } from './ReasoningPanel'

describe('ReasoningPanel', () => {
  afterEach(cleanup)

  it('renders "Reasoning" when thinking', () => {
    render(<ReasoningPanel isThinking={true} />)
    expect(screen.getByText('Reasoning')).toBeDefined()
  })

  it('renders "Reasoning complete" when not thinking', () => {
    render(<ReasoningPanel isThinking={false} />)
    expect(screen.getByText('Reasoning complete')).toBeDefined()
  })

  it('shows animated dots when thinking', () => {
    render(<ReasoningPanel isThinking={true} />)
    const dots = screen.getByLabelText('Show reasoning').querySelector('[aria-hidden="true"]')
    expect(dots).toBeDefined()
  })

  it('hides dots when not thinking', () => {
    render(<ReasoningPanel isThinking={false} />)
    const dots = screen.getByLabelText('Show reasoning').querySelector('[aria-hidden="true"]')
    expect(dots).toBeNull()
  })

  it('has aria-expanded false by default', () => {
    render(<ReasoningPanel isThinking={true} />)
    expect(screen.getByLabelText('Show reasoning').getAttribute('aria-expanded')).toBe('false')
  })

  it('toggles expanded state on click', () => {
    render(<ReasoningPanel isThinking={true} />)
    const btn = screen.getByLabelText('Show reasoning')
    fireEvent.click(btn)
    expect(btn.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(btn)
    expect(btn.getAttribute('aria-expanded')).toBe('false')
  })

  it('shows reasoning description when expanded and thinking', () => {
    render(<ReasoningPanel isThinking={true} />)
    fireEvent.click(screen.getByLabelText('Show reasoning'))
    expect(screen.getByText(/Generating response/)).toBeDefined()
  })

  it('hides description when not thinking even if expanded', () => {
    render(<ReasoningPanel isThinking={false} />)
    fireEvent.click(screen.getByLabelText('Show reasoning'))
    expect(screen.queryByText(/Generating response/)).toBeNull()
  })

  it('applies className', () => {
    const { container } = render(<ReasoningPanel isThinking={true} className="custom-class" />)
    expect(container.querySelector('.custom-class')).toBeDefined()
  })
})
