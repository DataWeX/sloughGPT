// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

import { LogConsole } from './LogConsole'

describe('LogConsole', () => {
  afterEach(cleanup)

  it('shows seed lines on mount', () => {
    render(<LogConsole />)
    expect(screen.getByText(/console online/)).toBeDefined()
    expect(screen.getByText(/session provider hydrated/)).toBeDefined()
    expect(screen.getByText(/API unreachable/)).toBeDefined()
  })

  it('shows line count in buffer', () => {
    render(<LogConsole />)
    expect(screen.getByText('buf:3')).toBeDefined()
  })

  it('shows tab items', () => {
    render(<LogConsole />)
    expect(screen.getByText('All')).toBeDefined()
    expect(screen.getByText('Info')).toBeDefined()
    expect(screen.getByText('Warn')).toBeDefined()
    expect(screen.getByText('Error')).toBeDefined()
  })

  it('filters by level', () => {
    render(<LogConsole />)
    const warnTab = screen.getByText('Warn')
    fireEvent.click(warnTab)
    expect(screen.getByText(/API unreachable/)).toBeDefined()
    expect(screen.queryByText(/console online/)).toBeNull()
  })

  it('shows empty state when filter matches nothing', () => {
    render(<LogConsole />)
    const errorTab = screen.getByText('Error')
    fireEvent.click(errorTab)
    expect(screen.getByText(/no frames for filter/i)).toBeDefined()
  })

  it('clear resets to seed lines', () => {
    render(<LogConsole />)
    const clearBtn = screen.getByLabelText('Clear log')
    fireEvent.click(clearBtn)
    expect(screen.getByText(/console online/)).toBeDefined()
  })

  it('appends line when tick changes', () => {
    const { rerender } = render(<LogConsole tick={0} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(3)
    rerender(<LogConsole tick={1} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(4)
    expect(screen.getByText(/poll\] metric/)).toBeDefined()
  })
})
