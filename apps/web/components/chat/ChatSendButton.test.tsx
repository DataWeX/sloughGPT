/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { ChatSendButton } from './ChatSendButton'

afterEach(cleanup)

describe('ChatSendButton', () => {
  it('renders send icon when not loading and hasContent', () => {
    render(<ChatSendButton loading={false} hasContent onSend={() => {}} disabled={false} />)
    const btn = screen.getByRole('button', { name: 'Send message' })
    expect(btn).toBeTruthy()
    expect(btn).not.toBeDisabled()
  })

  it('renders stop icon when loading', () => {
    render(<ChatSendButton loading hasContent onSend={() => {}} disabled={false} />)
    const btn = screen.getByRole('button', { name: 'Stop generation' })
    expect(btn).toBeTruthy()
  })

  it('is disabled when no content and not loading', () => {
    render(<ChatSendButton loading={false} hasContent={false} onSend={() => {}} disabled={false} />)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('is disabled when disabled prop is true', () => {
    render(<ChatSendButton loading={false} hasContent onSend={() => {}} disabled />)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('is NOT disabled when loading (so stop is clickable)', () => {
    render(<ChatSendButton loading hasContent onSend={() => {}} disabled />)
    expect(screen.getByRole('button')).not.toBeDisabled()
  })

  it('calls onSend on click when not loading', () => {
    const onSend = vi.fn()
    render(<ChatSendButton loading={false} hasContent onSend={onSend} disabled={false} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it('calls onStop on click when loading', () => {
    const onStop = vi.fn()
    render(<ChatSendButton loading hasContent onSend={() => {}} onStop={onStop} disabled={false} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onStop).toHaveBeenCalledTimes(1)
  })

  it('has data-send-button attribute', () => {
    render(<ChatSendButton loading={false} hasContent onSend={() => {}} disabled={false} />)
    expect(screen.getByRole('button').getAttribute('data-send-button')).toBe('true')
  })
})
