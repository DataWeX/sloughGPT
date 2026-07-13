'use client'

import {
  createContext,
  createRef,
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

/* ── Context ────────────────────────────────────────────────────── */

interface SelectContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  value: string
  onValueChange: (value: string) => void
  placeholder?: string
  triggerRef: React.RefObject<HTMLButtonElement | null>
  registerItem: (el: HTMLElement | null) => void
  unregisterItem: (el: HTMLElement | null) => void
  focusFirst: () => void
  focusNext: (current: HTMLElement) => void
  focusPrev: (current: HTMLElement) => void
}

const SelectContext = createContext<SelectContextValue | null>(null)

function useSelectContext() {
  const ctx = useContext(SelectContext)
  if (!ctx) throw new Error('Select compound components must be used within <Select>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface SelectRootProps {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  disabled?: boolean
  children: ReactNode
}

function Select({ value: controlledValue, defaultValue = '', onValueChange, children }: SelectRootProps) {
  const [internalValue, setInternalValue] = useState(defaultValue)
  const isControlled = controlledValue !== undefined
  const value = isControlled ? controlledValue : internalValue
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const itemsRef = useRef<Map<HTMLElement, number>>(new Map())
  const nextId = useRef(0)

  const handleChange = useCallback(
    (v: string) => {
      if (!isControlled) setInternalValue(v)
      onValueChange?.(v)
      setOpen(false)
    },
    [isControlled, onValueChange],
  )

  const registerItem = useCallback((el: HTMLElement | null) => {
    if (el && !itemsRef.current.has(el)) itemsRef.current.set(el, nextId.current++)
  }, [])

  const unregisterItem = useCallback((el: HTMLElement | null) => {
    if (el) itemsRef.current.delete(el)
  }, [])

  const sortedItems = useCallback(() => {
    return Array.from(itemsRef.current.entries())
      .sort((a, b) => a[1] - b[1])
      .map(([el]) => el)
  }, [])

  const focusFirst = useCallback(() => {
    const items = sortedItems()
    if (items.length > 0) items[0].focus()
  }, [sortedItems])

  const focusNext = useCallback(
    (current: HTMLElement) => {
      const items = sortedItems()
      const idx = items.indexOf(current)
      if (idx >= 0 && idx < items.length - 1) items[idx + 1].focus()
    },
    [sortedItems],
  )

  const focusPrev = useCallback(
    (current: HTMLElement) => {
      const items = sortedItems()
      const idx = items.indexOf(current)
      if (idx > 0) items[idx - 1].focus()
    },
    [sortedItems],
  )

  return (
    <SelectContext.Provider
      value={{ open, onOpenChange: setOpen, value, onValueChange: handleChange, triggerRef, registerItem, unregisterItem, focusFirst, focusNext, focusPrev }}
    >
      {children}
    </SelectContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────────── */

const SelectTrigger = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement> & { placeholder?: string }>(
  ({ className, children, placeholder, ...props }, ref) => {
    const ctx = useSelectContext()
    const mergedRef = useCallback(
      (node: HTMLButtonElement | null) => {
        ;(ctx.triggerRef as React.MutableRefObject<HTMLButtonElement | null>).current = node
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      },
      [ref, ctx.triggerRef],
    )

    return (
      <button
        ref={mergedRef}
        type="button"
        role="combobox"
        aria-expanded={ctx.open}
        aria-haspopup="listbox"
        onClick={() => ctx.onOpenChange(!ctx.open)}
        className={cn(
          'flex h-10 w-full items-center justify-between gap-2 rounded-lg border border-border bg-background px-3 py-2 text-sm',
          'focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className,
        )}
        {...props}
      >
        <span className={cn('truncate', !ctx.value && 'text-muted-foreground')}>
          {ctx.value ? children : placeholder ?? 'Select...'}
        </span>
        <svg className={cn('h-4 w-4 text-muted-foreground transition-transform', ctx.open && 'rotate-180')} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 9l6 6 6-6" />
        </svg>
      </button>
    )
  },
)
SelectTrigger.displayName = 'SelectTrigger'

/* ── Content ────────────────────────────────────────────────────── */

const SelectContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const ctx = useSelectContext()
    const contentRef = useRef<HTMLDivElement>(null)
    const previousActiveElement = useRef<HTMLElement | null>(null)

    useLayoutEffect(() => {
      if (ctx.open) {
        previousActiveElement.current = document.activeElement as HTMLElement
        requestAnimationFrame(() => ctx.focusFirst())
      } else if (previousActiveElement.current) {
        previousActiveElement.current.focus()
        previousActiveElement.current = null
      }
    }, [ctx.open, ctx.focusFirst])

    useEffect(() => {
      if (!ctx.open) return
      const handler = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          e.preventDefault()
          ctx.onOpenChange(false)
        } else if (e.key === 'ArrowDown') {
          e.preventDefault()
          ctx.focusNext(e.target as HTMLElement)
        } else if (e.key === 'ArrowUp') {
          e.preventDefault()
          ctx.focusPrev(e.target as HTMLElement)
        } else if (e.key === 'Home') {
          e.preventDefault()
          ctx.focusFirst()
        } else if (e.key === 'End') {
          e.preventDefault()
          const items = contentRef.current?.querySelectorAll<HTMLElement>('[role="option"]')
          items?.[items.length - 1]?.focus()
        }
      }
      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [ctx])

    useEffect(() => {
      if (!ctx.open) return
      const handler = (e: MouseEvent) => {
        const target = e.target as Node
        if (!contentRef.current?.contains(target) && !ctx.triggerRef.current?.contains(target)) {
          ctx.onOpenChange(false)
        }
      }
      document.addEventListener('mousedown', handler)
      return () => document.removeEventListener('mousedown', handler)
    }, [ctx])

    if (!ctx.open) return null

    return createPortal(
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="listbox"
        className={cn(
          'absolute z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-lg border border-border bg-popover text-popover-foreground shadow-lg',
          'animate-in fade-in-0 zoom-in-95 duration-150',
          className,
        )}
        {...props}
      >
        <div className="p-1">{children}</div>
      </div>,
      document.body,
    )
  },
)
SelectContent.displayName = 'SelectContent'

/* ── Item ───────────────────────────────────────────────────────── */

interface SelectItemProps extends HTMLAttributes<HTMLDivElement> {
  value: string
  disabled?: boolean
}

const SelectItem = forwardRef<HTMLDivElement, SelectItemProps>(
  ({ className, value: itemValue, disabled, children, ...props }, ref) => {
    const ctx = useSelectContext()
    const itemRef = useRef<HTMLDivElement>(null)
    const isSelected = ctx.value === itemValue

    useEffect(() => {
      ctx.registerItem(itemRef.current)
      return () => ctx.unregisterItem(itemRef.current)
    }, [ctx.registerItem, ctx.unregisterItem])

    return (
      <div
        ref={(node) => {
          ;(itemRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="option"
        aria-selected={isSelected}
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
        data-disabled={disabled ? '' : undefined}
        onClick={() => {
          if (!disabled) ctx.onValueChange(itemValue)
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !disabled) ctx.onValueChange(itemValue)
        }}
        className={cn(
          'relative flex w-full cursor-pointer select-none items-center gap-2 rounded-md py-1.5 pl-8 pr-2 text-sm outline-none',
          'focus:bg-primary/10 focus:text-foreground',
          'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
          className,
        )}
        {...props}
      >
        {isSelected && (
          <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
            <svg className="h-3.5 w-3.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          </span>
        )}
        {children}
      </div>
    )
  },
)
SelectItem.displayName = 'SelectItem'

/* ── Value ──────────────────────────────────────────────────────── */

function SelectValue({ placeholder }: { placeholder?: string }) {
  const ctx = useSelectContext()
  if (!ctx.value) return <span className="text-muted-foreground">{placeholder ?? 'Select...'}</span>
  return null
}

/* ── Group / Separator ──────────────────────────────────────────── */

const SelectGroup = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} role="group" className={className} {...props} />,
)
SelectGroup.displayName = 'SelectGroup'

const SelectSeparator = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} role="separator" className={cn('my-1 h-px bg-border', className)} {...props} />,
)
SelectSeparator.displayName = 'SelectSeparator'

export { Select, SelectTrigger, SelectContent, SelectItem, SelectValue, SelectGroup, SelectSeparator }
