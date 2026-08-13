'use client'

import {
  cloneElement,
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type HTMLAttributes,
  type MouseEvent as ReactMouseEvent,
  type ReactElement,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'

/* ── Context ────────────────────────────────────────────────────── */

interface DialogContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  titleId: string
  descriptionId: string
}

const DialogContext = createContext<DialogContextValue | null>(null)

function useDialogContext() {
  const ctx = useContext(DialogContext)
  if (!ctx) throw new Error('Dialog compound components must be used within <Dialog>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface DialogRootProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
}

function Dialog({ open: controlledOpen, defaultOpen = false, onOpenChange, children }: DialogRootProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen

  const setOpen = useCallback(
    (next: boolean) => {
      if (!isControlled) setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange],
  )

  const titleId = useId()
  const descriptionId = useId()

  return (
    <DialogContext.Provider value={{ open, onOpenChange: setOpen, titleId, descriptionId }}>
      {children}
    </DialogContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────────── */

const DialogTrigger = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement> & { asChild?: boolean }>(
  ({ onClick, asChild, children, ...props }, ref) => {
    const { onOpenChange } = useDialogContext()

    const triggerProps = {
      ref,
      type: 'button' as const,
      'aria-haspopup': 'dialog' as const,
      onClick: (e: ReactMouseEvent<HTMLButtonElement>) => {
        onClick?.(e)
        onOpenChange(true)
      },
    }

    if (asChild && children && typeof children === 'object' && 'props' in children) {
      const child = children as ReactElement
      return cloneElement(child, {
        ...triggerProps,
        ...child.props,
        ref,
        onClick: (e: ReactMouseEvent<HTMLButtonElement>) => {
          child.props.onClick?.(e)
          onClick?.(e)
          onOpenChange(true)
        },
      })
    }

    return (
      <button {...triggerProps} {...props}>
        {children}
      </button>
    )
  },
)
DialogTrigger.displayName = 'DialogTrigger'

/* ── Portal ─────────────────────────────────────────────────────── */

function DialogPortal({ children }: { children: ReactNode }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  if (!mounted) return null
  return createPortal(children, document.body)
}

/* ── Overlay ────────────────────────────────────────────────────── */

const DialogOverlay = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, onClick, ...props }, ref) => {
    const { open, onOpenChange } = useDialogContext()
    return (
      <div
        ref={ref}
        aria-hidden="true"
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(false)
        }}
        className={cn(
          'fixed inset-0 z-50 bg-foreground/20 backdrop-blur-sm transition-opacity duration-200',
          open ? 'opacity-100' : 'opacity-0 pointer-events-none',
          className,
        )}
        {...props}
      />
    )
  },
)
DialogOverlay.displayName = 'DialogOverlay'

/* ── Content ────────────────────────────────────────────────────── */

const DialogContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const { open, onOpenChange, titleId, descriptionId } = useDialogContext()
    const contentRef = useRef<HTMLDivElement>(null)
    const previousActiveElement = useRef<HTMLElement | null>(null)

    // Store previously focused element and restore on close
    useLayoutEffect(() => {
      if (open) {
        previousActiveElement.current = document.activeElement as HTMLElement
        // Focus the content
        requestAnimationFrame(() => {
          const focusable = contentRef.current?.querySelector<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
          )
          focusable?.focus()
        })
      } else if (previousActiveElement.current) {
        previousActiveElement.current.focus()
        previousActiveElement.current = null
      }
    }, [open])

    // Escape key
    useEffect(() => {
      if (!open) return
      const handler = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          e.preventDefault()
          onOpenChange(false)
        }
      }
      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [open, onOpenChange])

    // Focus trap
    useEffect(() => {
      if (!open) return
      const container = contentRef.current
      if (!container) return

      const handler = (e: KeyboardEvent) => {
        if (e.key !== 'Tab') return

        const focusable = container.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        )
        if (focusable.length === 0) return

        const first = focusable[0]
        const last = focusable[focusable.length - 1]

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault()
            last.focus()
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault()
            first.focus()
          }
        }
      }

      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [open])

    // Prevent body scroll when open
    useEffect(() => {
      if (!open) return
      const original = document.body.style.overflow
      document.body.style.overflow = 'hidden'
      return () => {
        document.body.style.overflow = original
      }
    }, [open])

    if (!open) return null

    return (
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        className={cn(
          'fixed left-[50%] top-[50%] z-50 grid w-[calc(100%-2rem)] max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border border-border bg-background p-6 text-foreground shadow-xl',
          'rounded-lg',
          'max-h-[min(90dvh,56.25rem)] overflow-y-auto',
          'animate-in fade-in-0 zoom-in-95 duration-200',
          className,
        )}
        {...props}
      >
        {children}
        <button
          type="button"
          aria-label="Close"
          className="absolute right-4 top-4 rounded-md opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2"
          onClick={() => onOpenChange(false)}
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          <span className="sr-only">Close</span>
        </button>
      </div>
    )
  },
)
DialogContent.displayName = 'DialogContent'

/* ── Close ──────────────────────────────────────────────────────── */

const DialogClose = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ onClick, ...props }, ref) => {
    const { onOpenChange } = useDialogContext()
    return (
      <button
        ref={ref}
        type="button"
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(false)
        }}
        {...props}
      />
    )
  },
)
DialogClose.displayName = 'DialogClose'

/* ── Layout helpers ─────────────────────────────────────────────── */

function DialogHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex flex-col gap-1.5 text-left', className)} {...props} />
}

function DialogFooter({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)} {...props} />
}

const DialogTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => {
    const { titleId } = useDialogContext()
    return <h2 ref={ref} id={titleId} className={cn('text-lg font-semibold leading-none tracking-tight', className)} {...props} />
  },
)
DialogTitle.displayName = 'DialogTitle'

const DialogDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => {
    const { descriptionId } = useDialogContext()
    return <p ref={ref} id={descriptionId} className={cn('text-sm text-muted-foreground', className)} {...props} />
  },
)
DialogDescription.displayName = 'DialogDescription'

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogClose,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
}
