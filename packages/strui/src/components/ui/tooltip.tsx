'use client'

import {
  cloneElement,
  createContext,
  forwardRef,
  isValidElement,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useId,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactElement,
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
      value={{
        open,
        onOpenChange: handleOpenChange,
        triggerRef,
        contentId: id,
        delay: delayDuration,
      }}
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

function composeHandlers<T extends Event>(
  first: ((e: T) => void) | undefined,
  second: ((e: T) => void) | undefined,
): ((e: T) => void) | undefined {
  if (!first) return second
  if (!second) return first
  return (e) => {
    first(e)
    second(e)
  }
}

/** Merge the trigger's event handlers/props onto a custom child element (Slot pattern). */
function mergeIntoChild(
  child: ReactElement<Record<string, unknown>>,
  ours: Record<string, unknown>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...ours, ...child.props }
  for (const key of Object.keys(ours)) {
    if (
      key.startsWith('on') &&
      typeof ours[key] === 'function' &&
      typeof child.props[key] === 'function'
    ) {
      merged[key] = composeHandlers(
        child.props[key] as (e: Event) => void,
        ours[key] as (e: Event) => void,
      )
    }
  }
  merged.className = cn(
    ours.className as string | undefined,
    child.props.className as string | undefined,
  )
  merged.style = { ...(ours.style as object), ...(child.props.style as object) }
  merged.ref = ours.ref ?? child.props.ref
  return merged
}

export const TooltipTrigger = forwardRef<HTMLButtonElement, TooltipTriggerProps>(
  ({ children, asChild = false, onMouseEnter, onMouseLeave, onFocus, onBlur, ...props }, ref) => {
    const { onOpenChange, triggerRef, contentId, delay } = useTooltipContext()
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const clearTimer = () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }

    const setTriggerRef = (node: HTMLElement | null) => {
      ;(triggerRef as React.MutableRefObject<HTMLElement | null>).current = node
      if (typeof ref === 'function') ref(node as HTMLButtonElement | null)
      else if (ref) ref.current = node as HTMLButtonElement | null
    }

    const triggerProps: Record<string, unknown> = {
      ref: setTriggerRef,
      type: 'button',
      'aria-describedby': contentId,
      onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
        timerRef.current = setTimeout(() => onOpenChange(true), delay)
        onMouseEnter?.(e)
      },
      onMouseLeave: (e: React.MouseEvent<HTMLElement>) => {
        clearTimer()
        onOpenChange(false)
        onMouseLeave?.(e)
      },
      onFocus: (e: React.FocusEvent<HTMLElement>) => {
        onOpenChange(true)
        onFocus?.(e)
      },
      onBlur: (e: React.FocusEvent<HTMLElement>) => {
        onOpenChange(false)
        onBlur?.(e)
      },
      ...props,
    }

    if (asChild && isValidElement(children)) {
      const child = children as ReactElement<Record<string, unknown>>
      return cloneElement(child, mergeIntoChild(child, triggerProps))
    }

    return (
      <button {...(triggerProps as React.ButtonHTMLAttributes<HTMLButtonElement>)}>
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

interface TooltipContentProps
  extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof tooltipContentVariants> {
  side?: Side
  align?: Align
  sideOffset?: number
}

export const TooltipContent = forwardRef<HTMLDivElement, TooltipContentProps>(
  (
    { className, children, variant, side = 'top', align = 'center', sideOffset = 8, ...props },
    ref,
  ) => {
    const { open, triggerRef, contentId } = useTooltipContext()
    const [mounted, setMounted] = useState(false)
    const [pos, setPos] = useState({ top: 0, left: 0 })
    const contentRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => setMounted(true), [])

    /* Position BEFORE paint (layout effect): a passive effect would render one
       frame at the mount-time {0,0} corner, and the `duration-150` class on the
       content would tween that correction (`transition-property` defaults to
       `all`), visibly sliding the tooltip in from the page corner. */
    useLayoutEffect(() => {
      if (!open || !triggerRef.current || !contentRef.current) return

      const trigger = triggerRef.current.getBoundingClientRect()
      const content = contentRef.current.getBoundingClientRect()
      const { scrollX, scrollY } = window

      /* Collision flip: if the preferred side can't fit the content and the
         opposite side has more room, resolve to the opposite side. */
      const room = {
        top: trigger.top - sideOffset,
        bottom: window.innerHeight - trigger.bottom - sideOffset,
        left: trigger.left - sideOffset,
        right: window.innerWidth - trigger.right - sideOffset,
      }
      let resolved = side
      if (side === 'top' && room.top < content.height && room.bottom > room.top) resolved = 'bottom'
      else if (side === 'bottom' && room.bottom < content.height && room.top > room.bottom)
        resolved = 'top'
      else if (side === 'left' && room.left < content.width && room.right > room.left)
        resolved = 'right'
      else if (side === 'right' && room.right < content.width && room.left > room.right)
        resolved = 'left'

      /* Align: cross-axis offset of content within the trigger's span. */
      const offsetX =
        align === 'start'
          ? 0
          : align === 'end'
            ? trigger.width - content.width
            : (trigger.width - content.width) / 2
      const offsetY =
        align === 'start'
          ? 0
          : align === 'end'
            ? trigger.height - content.height
            : (trigger.height - content.height) / 2

      let top = 0
      let left = 0

      if (resolved === 'top') {
        top = trigger.top + scrollY - content.height - sideOffset
        left = trigger.left + scrollX + offsetX
      } else if (resolved === 'bottom') {
        top = trigger.bottom + scrollY + sideOffset
        left = trigger.left + scrollX + offsetX
      } else if (resolved === 'left') {
        top = trigger.top + scrollY + offsetY
        left = trigger.left + scrollX - content.width - sideOffset
      } else {
        top = trigger.top + scrollY + offsetY
        left = trigger.right + scrollX + sideOffset
      }

      // Clamp to viewport
      left = Math.max(8, Math.min(left, window.innerWidth + scrollX - content.width - 8))
      top = Math.max(8, Math.min(top, window.innerHeight + scrollY - content.height - 8))

      setPos({ top, left })
    }, [open, mounted, side, align, sideOffset, triggerRef])

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
        /* transition:none overrides the base `duration-150` (whose default
           transition-property is `all`) so position changes never tween. The
           entry animation is unaffected — it runs on `animation`, not
           `transition`. Consumers can re-enable via props.style. */
        style={{ position: 'absolute', top: pos.top, left: pos.left, transition: 'none' }}
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
