// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { FeedbackFormCard } from './FeedbackFormCard'

afterEach(() => { cleanup() })

describe('FeedbackFormCard', () => {
  it('renders the form', () => {
    render(<FeedbackFormCard />)
    expect(screen.getAllByText('Submit Feedback').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('👍 Good')).toBeTruthy()
    expect(screen.getByText('👎 Bad')).toBeTruthy()
  })

  it('submit button disabled without rating', () => {
    render(<FeedbackFormCard />)
    const btn = screen.getByRole('button', { name: /Submit Feedback/ })
    expect(btn).toHaveProperty('disabled', true)
  })

  it('enables submit after selecting rating', () => {
    render(<FeedbackFormCard />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    const btn = screen.getByRole('button', { name: /Submit Feedback/ })
    expect(btn).not.toHaveProperty('disabled', true)
  })

  it('calls onSubmit with rating', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ rating: 'thumbs_up' }))
  })

  it('includes comment when provided', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs down'))
    fireEvent.change(screen.getByLabelText('Feedback comment'), { target: { value: 'Needs improvement' } })
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ comment: 'Needs improvement' }))
  })

  it('includes category when selected', () => {
    const onSubmit = vi.fn()
    render(<FeedbackFormCard onSubmit={onSubmit} />)
    fireEvent.click(screen.getByLabelText('Thumbs up'))
    fireEvent.click(screen.getByText('quality'))
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/ }))
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ category: 'quality' }))
  })

  it('shows categories', () => {
    render(<FeedbackFormCard />)
    expect(screen.getByText('quality')).toBeTruthy()
    expect(screen.getByText('accuracy')).toBeTruthy()
    expect(screen.getByText('helpfulness')).toBeTruthy()
    expect(screen.getByText('speed')).toBeTruthy()
    expect(screen.getByText('other')).toBeTruthy()
  })

  it('shows submitting state', () => {
    render(<FeedbackFormCard submitting />)
    const btn = screen.getByRole('button', { name: /Submitting/ })
    expect(btn).toHaveProperty('disabled', true)
  })
})
