import { renderToStaticMarkup } from 'react-dom/server'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  Toast,
  ToastAction,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  toastVariants,
  useToast,
} from './toast'

let toastCount = 0
let actionHandler = vi.fn()

function Harness() {
  const { toast, dismissAll } = useToast()
  return (
    <div>
      <button
        onClick={() => {
          toastCount += 1
          toast({ title: `Toast ${toastCount}`, description: `Desc ${toastCount}`, duration: 5000 })
        }}
      >
        Show toast
      </button>
      <button
        onClick={() => toast({ title: 'Action toast', action: { label: 'Undo', onClick: () => actionHandler() } })}
      >
        Show action
      </button>
      <button onClick={dismissAll}>Dismiss all</button>
    </div>
  )
}

function renderToast(maxToasts?: number) {
  return render(
    <ToastProvider maxToasts={maxToasts}>
      <Harness />
    </ToastProvider>,
  )
}

beforeEach(() => {
  toastCount = 0
  actionHandler = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('useToast + ToastProvider', () => {
  it('renders a toast with title and description through the portal', () => {
    renderToast()
    fireEvent.click(screen.getByText('Show toast'))
    expect(screen.getByText('Toast 1')).toBeTruthy()
    expect(screen.getByText('Desc 1')).toBeTruthy()
  })

  it('dismisses a toast via the dismiss button', () => {
    vi.useFakeTimers()
    renderToast()
    fireEvent.click(screen.getByText('Show toast'))
    fireEvent.click(screen.getByLabelText('Dismiss'))
    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(screen.queryByText('Toast 1')).toBeNull()
  })

  it('auto-dismisses after the configured duration', () => {
    vi.useFakeTimers()
    renderToast()
    fireEvent.click(screen.getByText('Show toast'))
    expect(screen.getByText('Toast 1')).toBeTruthy()
    act(() => {
      vi.advanceTimersByTime(6000)
    })
    expect(screen.queryByText('Toast 1')).toBeNull()
  })

  it('caps the toast stack at maxToasts', () => {
    vi.useFakeTimers()
    renderToast(3)
    for (let i = 0; i < 7; i += 1) {
      fireEvent.click(screen.getByText('Show toast'))
    }
    expect(screen.getAllByRole('status').length).toBe(3)
    expect(screen.queryByText('Toast 4')).toBeNull()
    act(() => {
      vi.advanceTimersByTime(10000)
    })
  })

  it('fires the action onClick and dismisses the toast', () => {
    vi.useFakeTimers()
    renderToast()
    fireEvent.click(screen.getByText('Show action'))
    fireEvent.click(screen.getByText('Undo'))
    expect(actionHandler).toHaveBeenCalledTimes(1)
    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(screen.queryByText('Action toast')).toBeNull()
  })

  it('dismissAll removes all toasts', () => {
    vi.useFakeTimers()
    renderToast()
    fireEvent.click(screen.getByText('Show toast'))
    fireEvent.click(screen.getByText('Show toast'))
    expect(screen.getAllByRole('status').length).toBe(2)
    fireEvent.click(screen.getByText('Dismiss all'))
    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(screen.queryAllByRole('status').length).toBe(0)
  })

  it('useToast throws outside the provider', () => {
    function Bad() {
      useToast()
      return null
    }
    expect(() => render(<Bad />)).toThrow('useToast must be used within <ToastProvider>')
  })
})

describe('toast primitives', () => {
  it('renders Toast with role status and children', () => {
    const html = renderToStaticMarkup(<Toast>Hello</Toast>)
    expect(html).toContain('role="status"')
    expect(html).toContain('Hello')
  })

  it('applies variant classes to Toast', () => {
    const html = renderToStaticMarkup(<Toast variant="success">Ok</Toast>)
    expect(html).toContain('bg-success/10')
  })

  it('renders ToastTitle, ToastDescription, ToastClose and ToastAction', () => {
    const html = renderToStaticMarkup(
      <div>
        <ToastTitle>Title</ToastTitle>
        <ToastDescription>Desc</ToastDescription>
        <ToastClose />
        <ToastAction>Go</ToastAction>
      </div>,
    )
    expect(html).toContain('Title')
    expect(html).toContain('Desc')
    expect(html).toContain('aria-label="Close"')
    expect(html).toContain('Go')
  })

  it('ToastClose fires onClick', () => {
    const onClick = vi.fn()
    render(<ToastClose onClick={onClick} />)
    fireEvent.click(screen.getByLabelText('Close'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('toastVariants returns variant classes', () => {
    expect(toastVariants({ variant: 'success' })).toContain('bg-success/10')
    expect(toastVariants({ variant: 'error' })).toContain('bg-destructive/10')
    expect(toastVariants({ variant: 'info' })).toContain('bg-accent')
    expect(toastVariants()).toContain('bg-background')
  })
})
