'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'
import { cva, type VariantProps } from 'class-variance-authority'

/* ─── Types ─────────────────────────────────────────────────── */

type Side = 'top' | 'right' | 'bottom' | 'left'
type Align = 'start' | 'center' | 'end'

/* ─── Context ───────────────────────────────────────────────── */

interface TooltipContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  triggerRef: React.RefObject<HTMLElement | null>
  contentId: string
  delay: number
}

const TooltipContext = createContext<TooltipContextValue | null>(null)

function useTooltipContext() {
  const ctx = useContext(TooltipContext)
  if (!ctx) throw new Error('Tooltip compound components must be used within <Tooltip>')
  return ctx
}

/* ─── Root ───────────────────────────────────────────────────── */

interface TooltipRootProps {
  children: ReactNode
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  /** Delay in ms before showing tooltip. Default: 400 */
  delayDuration?: number
}

export function Tooltip({
  children,
  open: controlledOpen,
  defaultOpen = false,
  onOpenChange,
  delayDuration = 400,
}: TooltipRootProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const triggerRef = useRef<HTMLElement | null>(null)
  const id = useId()

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!isControlled) setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange],
  )

  return (
    <TooltipContext.Provider
      value={{ open, onOpenChange: handleOpenChange, triggerRef, contentId: id, delay: delayDuration }}
    >
      {children}
    </TooltipContext.Provider>
  )
}

/* ─── Trigger ───────────────────────────────────────────────── */

interface TooltipTriggerProps extends HTMLAttributes<HTMLElement> {
  asChild?: boolean
  children: ReactNode
}

export const TooltipTrigger = forwardRef<HTMLButtonElement, TooltipTriggerProps>(
  ({ children, onMouseEnter, onMouseLeave, onFocus, onBlur, ...props }, ref) => {
    const { onOpenChange, triggerRef, contentId, delay } = useTooltipContext()
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const clearTimer = () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }

    return (
      <button
        ref={(node) => {
          ;(triggerRef as React.MutableRefObject<HTMLElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        type="button"
        aria-describedby={contentId}
        onMouseEnter={(e) => {
          timerRef.current = setTimeout(() => onOpenChange(true), delay)
          onMouseEnter?.(e as any)
        }}
        onMouseLeave={(e) => {
          clearTimer()
          onOpenChange(false)
          onMouseLeave?.(e as any)
        }}
        onFocus={(e) => {
          onOpenChange(true)
          onFocus?.(e as any)
        }}
        onBlur={(e) => {
          onOpenChange(false)
          onBlur?.(e as any)
        }}
        {...props}
      >
        {children}
      </button>
    )
  },
)
TooltipTrigger.displayName = 'TooltipTrigger'

/* ─── Content ───────────────────────────────────────────────── */

const tooltipContentVariants = cva(
  [
    'z-50 rounded-md px-2.5 py-1.5 text-xs font-medium shadow-md pointer-events-none',
    'animate-in fade-in-0 zoom-in-95 duration-150',
    'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
  ].join(' '),
  {
    variants: {
      variant: {
        default: 'bg-foreground text-background',
        muted: 'bg-card border border-border text-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  },
)

interface TooltipContentProps extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof tooltipContentVariants> {
  side?: Side
  align?: Align
  sideOffset?: number
}

export const TooltipContent = forwardRef<HTMLDivElement, TooltipContentProps>(
  ({ className, children, variant, side = 'top', align = 'center', sideOffset = 8, ...props }, ref) => {
    const { open, triggerRef, contentId } = useTooltipContext()
    const [mounted, setMounted] = useState(false)
    const [pos, setPos] = useState({ top: 0, left: 0 })
    const contentRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => setMounted(true), [])

    useEffect(() => {
      if (!open || !triggerRef.current || !contentRef.current) return

      const trigger = triggerRef.current.getBoundingClientRect()
      const content = contentRef.current.getBoundingClientRect()
      const { scrollX, scrollY } = window

      let top = 0
      let left = 0

      if (side === 'top') {
        top = trigger.top + scrollY - content.height - sideOffset
        left = trigger.left + scrollX + trigger.width / 2 - content.width / 2
      } else if (side === 'bottom') {
        top = trigger.bottom + scrollY + sideOffset
        left = trigger.left + scrollX + trigger.width / 2 - content.width / 2
      } else if (side === 'left') {
        top = trigger.top + scrollY + trigger.height / 2 - content.height / 2
        left = trigger.left + scrollX - content.width - sideOffset
      } else {
        top = trigger.top + scrollY + trigger.height / 2 - content.height / 2
        left = trigger.right + scrollX + sideOffset
      }

      // Clamp to viewport
      left = Math.max(8, Math.min(left, window.innerWidth + scrollX - content.width - 8))
      top = Math.max(8, Math.min(top, window.innerHeight + scrollY - content.height - 8))

      setPos({ top, left })
    }, [open, side, sideOffset, triggerRef])

    if (!mounted || !open) return null

    return createPortal(
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        id={contentId}
        role="tooltip"
        data-state={open ? 'open' : 'closed'}
        className={cn(tooltipContentVariants({ variant }), className)}
        style={{ position: 'absolute', top: pos.top, left: pos.left }}
        {...props}
      >
        {children}
      </div>,
      document.body,
    )
  },
)
TooltipContent.displayName = 'TooltipContent'

/* ─── Convenience: SimpleTooltip ────────────────────────────── */

interface SimpleTooltipProps {
  content: ReactNode
  side?: Side
  align?: Align
  delay?: number
  variant?: 'default' | 'muted'
  children: ReactNode
  className?: string
}

/**
 * Drop-in wrapper: wraps a single child in a Tooltip with no boilerplate.
 * Usage: <SimpleTooltip content="Copy"><Button /></SimpleTooltip>
 */
export function SimpleTooltip({
  content,
  side = 'top',
  align = 'center',
  delay = 400,
  variant = 'default',
  children,
  className,
}: SimpleTooltipProps) {
  return (
    <Tooltip delayDuration={delay}>
      <TooltipTrigger asChild className={className}>
        {children as any}
      </TooltipTrigger>
      <TooltipContent side={side} align={align} variant={variant}>
        {content}
      </TooltipContent>
    </Tooltip>
  )
}
