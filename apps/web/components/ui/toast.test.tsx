/**
 * Toast component tests — uses strui Toast primitives.
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ToastProvider, Toast, ToastTitle, ToastDescription, ToastClose, ToastAction } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Toast', () => {
  it('renders toast with title and description', () => {
    render(
      <ToastProvider>
        <Toast>
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
        <Toast>
          <ToastTitle>Update available</ToastTitle>
          <ToastAction aria-label="Install update">Install</ToastAction>
          <ToastClose />
        </Toast>
      </ToastProvider>
    )
    expect(screen.getByText('Install')).toBeInTheDocument()
  })

  it('renders with variant styling', () => {
    const { container } = render(
      <ToastProvider>
        <Toast variant="error">
          <ToastTitle>Error</ToastTitle>
        </Toast>
      </ToastProvider>
    )
    const toast = container.querySelector('[role="status"]')
    expect(toast).toBeTruthy()
  })
})
