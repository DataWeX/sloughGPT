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

  it('renders after loading', async () => {
    await act(async () => {
      render(<ConversationTags sessionId="s1" />)
    })
    expect(screen.getByPlaceholderText('Add tag...')).toBeInTheDocument()
  })

  it('applies custom className', async () => {
    let container: ReturnType<typeof render>['container']
    await act(async () => {
      const result = render(<ConversationTags sessionId="s1" className="custom" />)
      container = result.container
    })
    expect(container!.firstChild).toHaveClass('custom')
  })
})
