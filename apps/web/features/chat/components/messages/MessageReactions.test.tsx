import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/dev-log', () => ({
  logger: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

import { MessageReactions } from './MessageReactions'

const defaultProps = {
  onReact: vi.fn(),
  reactions: { '👍': 3, '❤️': 1, '🔥': 2 },
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(cleanup)

describe('MessageReactions', () => {
  it('renders without crashing', () => {
    render(<MessageReactions {...defaultProps} />)
    expect(screen.getByText('👍')).toBeDefined()
    expect(screen.getByText('❤️')).toBeDefined()
    expect(screen.getByText('🔥')).toBeDefined()
  })

  it('displays reaction counts for multi-count reactions', () => {
    render(<MessageReactions {...defaultProps} />)
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
    expect(screen.queryByText('1')).toBeNull()
  })

  it('hides count when reaction count is 1', () => {
    render(<MessageReactions {...defaultProps} />)
    const heartBtn = screen.getByLabelText(/❤️/)
    expect(heartBtn.textContent).toBe('❤️')
  })

  it('calls onReact when clicking an existing reaction', () => {
    const onReact = vi.fn()
    render(<MessageReactions {...defaultProps} onReact={onReact} />)
    fireEvent.click(screen.getByLabelText(/👍/))
    expect(onReact).toHaveBeenCalledWith('👍')
  })

  it('opens picker when add reaction button clicked', () => {
    render(<MessageReactions {...defaultProps} />)
    const addBtn = screen.getByLabelText('Add reaction')
    fireEvent.click(addBtn)
    expect(screen.getByLabelText(/React with 👍/)).toBeDefined()
    expect(screen.getByLabelText(/React with ❤️/)).toBeDefined()
    expect(screen.getByLabelText(/React with 😊/)).toBeDefined()
  })

  it('calls onReact and closes picker when selecting from picker', () => {
    const onReact = vi.fn()
    render(<MessageReactions {...defaultProps} onReact={onReact} />)
    fireEvent.click(screen.getByLabelText('Add reaction'))
    fireEvent.click(screen.getByLabelText(/React with 🤔/))
    expect(onReact).toHaveBeenCalledWith('🤔')
    expect(screen.queryByLabelText(/React with 👍/)).toBeNull()
  })

  it('closes picker on Escape key', () => {
    render(<MessageReactions {...defaultProps} />)
    fireEvent.click(screen.getByLabelText('Add reaction'))
    expect(screen.getByLabelText(/React with 👍/)).toBeDefined()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByLabelText(/React with 👍/)).toBeNull()
  })

  it('closes picker on outside click', () => {
    render(<MessageReactions {...defaultProps} />)
    fireEvent.click(screen.getByLabelText('Add reaction'))
    expect(screen.getByLabelText(/React with 👍/)).toBeDefined()
    fireEvent.mouseDown(document.body)
    expect(screen.queryByLabelText(/React with 👍/)).toBeNull()
  })

  it('renders empty reactions gracefully', () => {
    render(<MessageReactions reactions={{}} onReact={vi.fn()} />)
    expect(screen.getByLabelText('Add reaction')).toBeDefined()
  })

  it('filters out reactions with zero count', () => {
    render(<MessageReactions reactions={{ '👍': 0, '❤️': 2 }} onReact={vi.fn()} />)
    expect(screen.queryByLabelText(/👍/)).toBeNull()
    expect(screen.getByLabelText(/❤️/)).toBeDefined()
  })

  it('applies custom className', () => {
    render(<MessageReactions {...defaultProps} className="custom-class" />)
    expect(screen.getByText('👍')).toBeDefined()
  })
})
