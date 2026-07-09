'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'

/* ─── Types ─────────────────────────────────────────────────── */

type Side = 'top' | 'right' | 'bottom' | 'left'
type Align = 'start' | 'center' | 'end'

/* ─── Context ───────────────────────────────────────────────── */

interface PopoverContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  triggerRef: React.RefObject<HTMLElement | null>
}

const PopoverContext = createContext<PopoverContextValue | null>(null)

function usePopoverContext() {
  const ctx = useContext(PopoverContext)
  if (!ctx) throw new Error('Popover compound components must be used within <Popover>')
  return ctx
}

/* ─── Root ───────────────────────────────────────────────────── */

interface PopoverRootProps {
  children: ReactNode
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
}

export function Popover({ children, open: controlledOpen, defaultOpen = false, onOpenChange }: PopoverRootProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const triggerRef = useRef<HTMLElement | null>(null)

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!isControlled) setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange],
  )

  return (
    <PopoverContext.Provider value={{ open, onOpenChange: handleOpenChange, triggerRef }}>
      {children}
    </PopoverContext.Provider>
  )
}

/* ─── Trigger ───────────────────────────────────────────────── */

export const PopoverTrigger = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement> & { asChild?: boolean }>(
  ({ onClick, children, ...props }, ref) => {
    const { onOpenChange, open, triggerRef } = usePopoverContext()

    return (
      <button
        ref={(node) => {
          ;(triggerRef as React.MutableRefObject<HTMLElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        type="button"
        aria-haspopup="dialog"
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
PopoverTrigger.displayName = 'PopoverTrigger'

/* ─── Content ───────────────────────────────────────────────── */

interface PopoverContentProps extends HTMLAttributes<HTMLDivElement> {
  side?: Side
  align?: Align
  sideOffset?: number
  /** Close when clicking outside. Default: true */
  closeOnOutsideClick?: boolean
}

export const PopoverContent = forwardRef<HTMLDivElement, PopoverContentProps>(
  (
    {
      className,
      children,
      side = 'bottom',
      align = 'center',
      sideOffset = 8,
      closeOnOutsideClick = true,
      ...props
    },
    ref,
  ) => {
    const { open, onOpenChange, triggerRef } = usePopoverContext()
    const [mounted, setMounted] = useState(false)
    const [pos, setPos] = useState({ top: 0, left: 0 })
    const contentRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => setMounted(true), [])

    // Position calculation
    useEffect(() => {
      if (!open || !triggerRef.current || !contentRef.current) return

      const trigger = triggerRef.current.getBoundingClientRect()
      const content = contentRef.current.getBoundingClientRect()
      const { scrollX, scrollY } = window

      let top = 0
      let left = 0

      if (side === 'bottom') {
        top = trigger.bottom + scrollY + sideOffset
        left =
          align === 'start'
            ? trigger.left + scrollX
            : align === 'end'
              ? trigger.right + scrollX - content.width
              : trigger.left + scrollX + trigger.width / 2 - content.width / 2
      } else if (side === 'top') {
        top = trigger.top + scrollY - content.height - sideOffset
        left =
          align === 'start'
            ? trigger.left + scrollX
            : align === 'end'
              ? trigger.right + scrollX - content.width
              : trigger.left + scrollX + trigger.width / 2 - content.width / 2
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
    }, [open, side, align, sideOffset, triggerRef])

    // Close on outside click
    useEffect(() => {
      if (!open || !closeOnOutsideClick) return
      const handler = (e: MouseEvent) => {
        const target = e.target as Node
        if (
          contentRef.current?.contains(target) ||
          triggerRef.current?.contains(target)
        ) return
        onOpenChange(false)
      }
      document.addEventListener('mousedown', handler)
      return () => document.removeEventListener('mousedown', handler)
    }, [open, closeOnOutsideClick, onOpenChange, triggerRef])

    // Close on Escape
    useEffect(() => {
      if (!open) return
      const handler = (e: KeyboardEvent) => {
        if (e.key === 'Escape') onOpenChange(false)
      }
      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [open, onOpenChange])

    if (!mounted || !open) return null

    return createPortal(
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="dialog"
        aria-modal="false"
        data-state={open ? 'open' : 'closed'}
        className={cn(
          'z-50 min-w-[8rem] overflow-hidden rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-lg',
          'animate-in fade-in-0 zoom-in-95 duration-150',
          className,
        )}
        style={{ position: 'absolute', top: pos.top, left: pos.left }}
        {...props}
      >
        {children}
      </div>,
      document.body,
    )
  },
)
PopoverContent.displayName = 'PopoverContent'

/* ─── Close ─────────────────────────────────────────────────── */

export const PopoverClose = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ onClick, children, ...props }, ref) => {
    const { onOpenChange } = usePopoverContext()
    return (
      <button
        ref={ref}
        type="button"
        onClick={(e) => {
          onClick?.(e)
          onOpenChange(false)
        }}
        {...props}
      >
        {children}
      </button>
    )
  },
)
PopoverClose.displayName = 'PopoverClose'

/* ─── Anchor ─────────────────────────────────────────────────── */

export function PopoverAnchor({ children }: { children: ReactNode }) {
  return <>{children}</>
}
