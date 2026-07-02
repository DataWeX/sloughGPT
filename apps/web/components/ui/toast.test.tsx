/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ToastProvider, Toast, ToastTitle, ToastDescription, ToastClose, ToastAction } from './toast'

afterEach(cleanup)

describe('Toast', () => {
  it('renders toast with title and description', () => {
    render(
      <ToastProvider>
        <Toast open>
          <ToastTitle>Success</ToastTitle>
          <ToastDescription>Operation completed</ToastDescription>
          <ToastClose />
        </Toast>
      </ToastProvider>
    )
    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('Operation completed')).toBeInTheDocument()
  })

  it('renders toast with action', () => {
    render(
      <ToastProvider>
        <Toast open>
          <ToastTitle>Update available</ToastTitle>
          <ToastAction aria-label="Install update">Install</ToastAction>
          <ToastClose />
        </Toast>
      </ToastProvider>
    )
    expect(screen.getByText('Install')).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    const { container } = render(
      <ToastProvider>
        <Toast open={false}>
          <ToastTitle>Hidden</ToastTitle>
        </Toast>
      </ToastProvider>
    )
    expect(container.textContent).toBe('')
  })
})
