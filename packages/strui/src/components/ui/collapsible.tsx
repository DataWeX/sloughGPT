'use client'

import { createContext, forwardRef, useCallback, useContext, useState, type HTMLAttributes, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

/* ── Context ────────────────────────────────────────────────────── */

interface CollapsibleContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const CollapsibleContext = createContext<CollapsibleContextValue | null>(null)

function useCollapsibleContext() {
  const ctx = useContext(CollapsibleContext)
  if (!ctx) throw new Error('Collapsible compound components must be used within <Collapsible>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface CollapsibleRootProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
  className?: string
}

function Collapsible({ open: controlledOpen, defaultOpen = false, onOpenChange, children, className }: CollapsibleRootProps) {
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

  return (
    <CollapsibleContext.Provider value={{ open, onOpenChange: setOpen }}>
      <div className={className}>{children}</div>
    </CollapsibleContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────────── */

const CollapsibleTrigger = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ onClick, children, ...props }, ref) => {
    const { open, onOpenChange } = useCollapsibleContext()
    return (
      <button
        ref={ref}
        type="button"
        aria-expanded={open}
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(!open)
        }}
        {...props}
      >
        {children}
      </button>
    )
  },
)
CollapsibleTrigger.displayName = 'CollapsibleTrigger'

/* ── Content ────────────────────────────────────────────────────── */

const CollapsibleContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const { open } = useCollapsibleContext()
    if (!open) return null
    return (
      <div ref={ref} role="region" className={cn('overflow-hidden', className)} {...props}>
        {children}
      </div>
    )
  },
)
CollapsibleContent.displayName = 'CollapsibleContent'

export { Collapsible, CollapsibleTrigger, CollapsibleContent }
