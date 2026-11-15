/**
 * MessageFrame — the one surface every chat message renders through.
 *
 * Direction B (user ruling 2026-11-03, "visual consistency please"): both
 * roles share one shape and radius; only alignment and tint differ, and the
 * metadata footer lives INSIDE the frame (no detached right gutter). These
 * tests are the mechanical anti-drift check the card asks for — renderers
 * must compose this frame, never re-decide chrome.
 */
import { describe, expect, it, afterEach } from 'vitest'
import { cleanup, render } from '@testing-library/react'
import { MessageFrame } from './MessageFrame'

afterEach(cleanup)

function frameOf(container: HTMLElement): HTMLElement {
  const frame = container.querySelector('[data-message-frame]')
  expect(frame).not.toBeNull()
  return frame as HTMLElement
}

function surfaceOf(container: HTMLElement): HTMLElement {
  return frameOf(container).firstElementChild as HTMLElement
}

describe('MessageFrame', () => {
  it('renders the user role right-aligned on a primary-tinted surface', () => {
    const { container } = render(
      <MessageFrame role="user">
        <p>hello</p>
      </MessageFrame>,
    )
    expect(frameOf(container)).toHaveAttribute('data-message-frame', 'user')
    expect(frameOf(container).className).toContain('justify-end')
    const surface = surfaceOf(container)
    expect(surface.className).toContain('bg-primary/10')
    expect(surface.className).toContain('border-primary/20')
  })

  it('renders the assistant role left-aligned on a card surface', () => {
    const { container } = render(
      <MessageFrame role="assistant">
        <p>hi</p>
      </MessageFrame>,
    )
    expect(frameOf(container)).toHaveAttribute('data-message-frame', 'assistant')
    expect(frameOf(container).className).not.toContain('justify-end')
    const surface = surfaceOf(container)
    expect(surface.className).toContain('bg-card')
    expect(surface.className).toContain('border-border')
  })

  it('gives both roles the same radius — one shape, no pills', () => {
    const user = render(
      <MessageFrame role="user">
        <p>q</p>
      </MessageFrame>,
    )
    const userSurface = surfaceOf(user.container)
    cleanup()
    const assistant = render(
      <MessageFrame role="assistant">
        <p>a</p>
      </MessageFrame>,
    )
    const assistantSurface = surfaceOf(assistant.container)
    expect(userSurface.className).toContain('rounded-2xl')
    expect(assistantSurface.className).toContain('rounded-2xl')
    expect(userSurface.className).not.toContain('rounded-full')
    expect(assistantSurface.className).not.toContain('rounded-full')
  })

  it('keeps the metadata footer inside the frame', () => {
    const { container } = render(
      <MessageFrame role="assistant" footer={<span>42 tok</span>}>
        <p>body</p>
      </MessageFrame>,
    )
    const frame = frameOf(container)
    const footer = frame.querySelector('[data-message-frame-footer]')
    expect(footer).not.toBeNull()
    expect(footer!.textContent).toContain('42 tok')
    expect(frame.textContent).toContain('body')
  })

  it('omits the footer slot when no footer is given', () => {
    const { container } = render(
      <MessageFrame role="user">
        <p>x</p>
      </MessageFrame>,
    )
    expect(container.querySelector('[data-message-frame-footer]')).toBeNull()
  })
})
