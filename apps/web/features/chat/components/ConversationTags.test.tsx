import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ConversationTags } from './ConversationTags'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

describe('ConversationTags', () => {
  it('renders tag input', async () => {
    await act(async () => {
      render(<ConversationTags sessionId="s1" />)
    })
    expect(screen.getByPlaceholderText('Add tag...')).toBeInTheDocument()
  })

  it('shows nothing while loading', () => {
    const { container } = render(<ConversationTags sessionId="s1" />)
    expect(container.firstChild).toBeNull()
  })

  it('applies custom className', async () => {
    await act(async () => {
      const { container } = render(<ConversationTags sessionId="s1" className="custom" />)
      expect(container.firstChild).toHaveClass('custom')
    })
  })
})
