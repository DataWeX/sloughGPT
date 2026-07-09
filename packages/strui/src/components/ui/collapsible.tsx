'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { cn } from '../../lib/cn'

/* ── Context ────────────────────────────────────────────────── */

interface CollapsibleContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  disabled?: boolean
}

const CollapsibleContext = createContext<CollapsibleContextValue | null>(null)

function useCollapsibleContext() {
  const ctx = useContext(CollapsibleContext)
  if (!ctx) throw new Error('Collapsible compound components must be used within <Collapsible>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────── */

interface CollapsibleRootProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
  className?: string
  disabled?: boolean
}

function Collapsible({
  open: controlledOpen,
  defaultOpen = false,
  onOpenChange,
  children,
  className,
  disabled,
}: CollapsibleRootProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen

  const setOpen = useCallback(
    (next: boolean) => {
      if (disabled) return
      if (!isControlled) setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange, disabled],
  )

  return (
    <CollapsibleContext.Provider value={{ open, onOpenChange: setOpen, disabled }}>
      <div className={className} data-state={open ? 'open' : 'closed'}>
        {children}
      </div>
    </CollapsibleContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────── */

const CollapsibleTrigger = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ onClick, children, className, ...props }, ref) => {
    const { open, onOpenChange, disabled } = useCollapsibleContext()
    return (
      <button
        ref={ref}
        type="button"
        aria-expanded={open}
        disabled={disabled}
        data-state={open ? 'open' : 'closed'}
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(!open)
        }}
        className={cn(
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          className,
        )}
        {...props}
      >
        {children}
      </button>
    )
  },
)
CollapsibleTrigger.displayName = 'CollapsibleTrigger'

/* ── Content ────────────────────────────────────────────────── */

const CollapsibleContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const { open } = useCollapsibleContext()

    // CSS-driven animation (no JS RAF needed — uses grid-template-rows trick)
    return (
      <div
        ref={ref}
        role="region"
        data-state={open ? 'open' : 'closed'}
        className={cn(
          'grid transition-[grid-template-rows] duration-200 ease-smooth',
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
          className,
        )}
        {...props}
      >
        <div className="overflow-hidden">{children}</div>
      </div>
    )
  },
)
CollapsibleContent.displayName = 'CollapsibleContent'

export { Collapsible, CollapsibleTrigger, CollapsibleContent }
