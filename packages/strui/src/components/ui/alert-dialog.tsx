'use client'

import React, {
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
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'
import { buttonVariants } from './button'

/* ── Context ────────────────────────────────────────────────────── */

interface AlertDialogContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  titleId: string
  descriptionId: string
}

const AlertDialogContext = createContext<AlertDialogContextValue | null>(null)

function useAlertDialogContext() {
  const ctx = useContext(AlertDialogContext)
  if (!ctx) throw new Error('AlertDialog compound components must be used within <AlertDialog>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface AlertDialogRootProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
}

function AlertDialog({ open: controlledOpen, defaultOpen = false, onOpenChange, children }: AlertDialogRootProps) {
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
    <AlertDialogContext.Provider value={{ open, onOpenChange: setOpen, titleId, descriptionId }}>
      {children}
    </AlertDialogContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────────── */

interface AlertDialogTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  asChild?: boolean
}

const AlertDialogTrigger = forwardRef<HTMLButtonElement, AlertDialogTriggerProps>(
  ({ onClick, asChild, children, ...props }, ref) => {
    const { onOpenChange } = useAlertDialogContext()

    const triggerProps = {
      ref,
      type: 'button' as const,
      'aria-haspopup': 'dialog' as const,
      onClick: (e: React.MouseEvent) => {
        onClick?.(e as any)
        onOpenChange(true)
      },
    }

    if (asChild && children && typeof children === 'object' && 'props' in children) {
      const child = children as React.ReactElement
      return React.cloneElement(child, {
        ...triggerProps,
        ...child.props,
        ref,
        onClick: (e: React.MouseEvent) => {
          child.props.onClick?.(e)
          onClick?.(e as any)
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
AlertDialogTrigger.displayName = 'AlertDialogTrigger'

/* ── Portal ─────────────────────────────────────────────────────── */

function AlertDialogPortal({ children }: { children: ReactNode }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  if (!mounted) return null
  return createPortal(children, document.body)
}

/* ── Overlay ────────────────────────────────────────────────────── */

const AlertDialogOverlay = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => {
    const { open } = useAlertDialogContext()
    return (
      <div
        ref={ref}
        aria-hidden="true"
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
AlertDialogOverlay.displayName = 'AlertDialogOverlay'

/* ── Content ────────────────────────────────────────────────────── */

const AlertDialogContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const { open, onOpenChange, titleId, descriptionId } = useAlertDialogContext()
    const contentRef = useRef<HTMLDivElement>(null)
    const previousActiveElement = useRef<HTMLElement | null>(null)

    useLayoutEffect(() => {
      if (open) {
        previousActiveElement.current = document.activeElement as HTMLElement
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
          if (document.activeElement === first) { e.preventDefault(); last.focus() }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus() }
        }
      }
      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [open])

    useEffect(() => {
      if (!open) return
      const original = document.body.style.overflow
      document.body.style.overflow = 'hidden'
      return () => { document.body.style.overflow = original }
    }, [open])

    if (!open) return null

    return (
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="alertdialog"
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
      </div>
    )
  },
)
AlertDialogContent.displayName = 'AlertDialogContent'

/* ── Layout helpers ─────────────────────────────────────────────── */

const AlertDialogHeader = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex flex-col gap-1.5 text-left', className)} {...props} />
)

const AlertDialogFooter = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)} {...props} />
)

const AlertDialogTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => {
    const { titleId } = useAlertDialogContext()
    return <h2 ref={ref} id={titleId} className={cn('text-lg font-semibold text-foreground', className)} {...props} />
  },
)
AlertDialogTitle.displayName = 'AlertDialogTitle'

const AlertDialogDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => {
    const { descriptionId } = useAlertDialogContext()
    return <p ref={ref} id={descriptionId} className={cn('text-sm text-muted-foreground', className)} {...props} />
  },
)
AlertDialogDescription.displayName = 'AlertDialogDescription'

/* ── Action / Cancel ────────────────────────────────────────────── */

const AlertDialogAction = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ className, onClick, ...props }, ref) => {
    const { onOpenChange } = useAlertDialogContext()
    return (
      <button
        ref={ref}
        type="button"
        className={cn(buttonVariants(), className)}
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(false)
        }}
        {...props}
      />
    )
  },
)
AlertDialogAction.displayName = 'AlertDialogAction'

const AlertDialogCancel = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ className, onClick, ...props }, ref) => {
    const { onOpenChange } = useAlertDialogContext()
    return (
      <button
        ref={ref}
        type="button"
        className={cn(buttonVariants({ variant: 'secondary' }), className)}
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(false)
        }}
        {...props}
      />
    )
  },
)
AlertDialogCancel.displayName = 'AlertDialogCancel'

export {
  AlertDialog,
  AlertDialogPortal,
  AlertDialogOverlay,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogAction,
  AlertDialogCancel,
}
