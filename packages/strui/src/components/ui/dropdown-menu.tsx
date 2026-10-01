'use client'

import React, {
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

interface DropdownMenuContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  triggerRef: React.RefObject<HTMLButtonElement | null>
  registerItem: (el: HTMLElement | null) => void
  unregisterItem: (el: HTMLElement | null) => void
  focusFirst: () => void
  focusLast: () => void
  focusNext: (current: HTMLElement) => void
  focusPrev: (current: HTMLElement) => void
}

const DropdownMenuContext = createContext<DropdownMenuContextValue | null>(null)

function useDropdownMenuContext() {
  const ctx = useContext(DropdownMenuContext)
  if (!ctx) throw new Error('DropdownMenu compound components must be used within <DropdownMenu>')
  return ctx
}

/* ── Root ───────────────────────────────────────────────────────── */

interface DropdownMenuRootProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
}

function DropdownMenu({ open: controlledOpen, defaultOpen = false, onOpenChange, children }: DropdownMenuRootProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const itemsRef = useRef<Map<HTMLElement, number>>(new Map())
  const nextId = useRef(0)

  const setOpen = useCallback(
    (next: boolean) => {
      if (!isControlled) setInternalOpen(next)
      onOpenChange?.(next)
    },
    [isControlled, onOpenChange],
  )

  const registerItem = useCallback((el: HTMLElement | null) => {
    if (el && !itemsRef.current.has(el)) {
      itemsRef.current.set(el, nextId.current++)
    }
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

  const focusLast = useCallback(() => {
    const items = sortedItems()
    if (items.length > 0) items[items.length - 1].focus()
  }, [sortedItems])

  const focusNext = useCallback(
    (current: HTMLElement) => {
      const items = sortedItems()
      const idx = items.indexOf(current)
      if (idx >= 0 && idx < items.length - 1) {
        items[idx + 1].focus()
      } else if (idx === -1) {
        focusFirst()
      }
    },
    [sortedItems, focusFirst],
  )

  const focusPrev = useCallback(
    (current: HTMLElement) => {
      const items = sortedItems()
      const idx = items.indexOf(current)
      if (idx > 0) {
        items[idx - 1].focus()
      } else if (idx === -1) {
        focusLast()
      }
    },
    [sortedItems, focusLast],
  )

  return (
    <DropdownMenuContext.Provider
      value={{ open, onOpenChange: setOpen, triggerRef, registerItem, unregisterItem, focusFirst, focusLast, focusNext, focusPrev }}
    >
      {children}
    </DropdownMenuContext.Provider>
  )
}

/* ── Trigger ────────────────────────────────────────────────────── */

interface DropdownMenuTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  asChild?: boolean
}

const DropdownMenuTrigger = forwardRef<HTMLButtonElement, DropdownMenuTriggerProps>(
  ({ onClick, asChild, children, ...props }, ref) => {
    const { open, onOpenChange, triggerRef } = useDropdownMenuContext()
    const mergedRef = useCallback(
      (node: HTMLButtonElement | null) => {
        ;(triggerRef as React.MutableRefObject<HTMLButtonElement | null>).current = node
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      },
      [ref, triggerRef],
    )

    const triggerProps = {
      ref: mergedRef,
      type: 'button' as const,
      'aria-haspopup': 'menu' as const,
      'aria-expanded': open,
      onClick: (e: React.MouseEvent) => {
        onClick?.(e as any)
        onOpenChange(!open)
      },
    }

    if (asChild && children && typeof children === 'object' && 'props' in children) {
      const child = children as React.ReactElement<Record<string, unknown>>
      const childProps = child.props as Record<string, unknown>
      const childOnClick = childProps.onClick as ((e: React.MouseEvent) => void) | undefined
      return React.cloneElement(child, {
        ...triggerProps,
        ...childProps,
        ref: mergedRef,
        onClick: (e: React.MouseEvent) => {
          childOnClick?.(e)
          onClick?.(e as any)
          onOpenChange(!open)
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
DropdownMenuTrigger.displayName = 'DropdownMenuTrigger'

/* ── Content ────────────────────────────────────────────────────── */

interface DropdownMenuContentProps extends HTMLAttributes<HTMLDivElement> {
  align?: 'start' | 'center' | 'end'
  sideOffset?: number
}

type MenuAnchor = { position: 'fixed'; top: number; left: number }

const DropdownMenuContent = forwardRef<HTMLDivElement, DropdownMenuContentProps>(
  ({ className, align = 'end', sideOffset = 6, children, style, ...props }, ref) => {
    const { open, onOpenChange, triggerRef, focusFirst, focusNext, focusPrev } = useDropdownMenuContext()
    const contentRef = useRef<HTMLDivElement>(null)
    /** Bounds the re-measure loop when width and `left` chase each other. */
    const corrections = useRef(0)
    const previousActiveElement = useRef<HTMLElement | null>(null)
    const [anchor, setAnchor] = useState<MenuAnchor | null>(null)

    /**
     * Anchor the menu to its trigger instead of letting it land wherever the
     * portal happens to sit in document flow. Flips above the trigger when
     * there is no room below, and clamps inside the viewport.
     */
    const place = useCallback((): MenuAnchor | null => {
      const trigger = triggerRef.current
      const content = contentRef.current
      if (!trigger || !content) return null
      const t = trigger.getBoundingClientRect()
      // offsetWidth/offsetHeight are layout sizes. The menu is still running
      // `zoom-in-95`, so getBoundingClientRect reports the scaled box — using
      // it for the width overshoots the alignment by the 5% deficit.
      const width = content.offsetWidth
      const height = content.offsetHeight
      const margin = 8
      let left =
        align === 'start'
          ? t.left
          : align === 'center'
            ? t.left + (t.width - width) / 2
            : t.right - width
      left = Math.min(Math.max(margin, left), Math.max(margin, window.innerWidth - width - margin))
      let top = t.bottom + sideOffset
      if (top + height > window.innerHeight - margin) {
        const above = t.top - sideOffset - height
        top = above >= margin ? above : Math.max(margin, window.innerHeight - height - margin)
      }
      return { position: 'fixed', top: Math.round(top), left: Math.round(left) }
    }, [align, sideOffset, triggerRef])

    /**
     * Place once on open, then keep watching: the menu's own box changes after
     * the first placement (it goes from "as wide as <body>" to shrink-to-fit),
     * so a single measurement settles against a stale rectangle and clamps `left`
     * to the viewport floor.
     */
    useLayoutEffect(() => {
      if (!open) {
        setAnchor(null)
        corrections.current = 0
        return
      }
      let active = true
      const apply = () => {
        if (!active) return
        const next = place()
        if (!next) return
        setAnchor((prev) =>
          prev && prev.top === next.top && prev.left === next.left ? prev : next,
        )
      }
      apply()

      window.addEventListener('resize', apply)
      window.addEventListener('scroll', apply, true)
      const observer =
        typeof ResizeObserver !== 'undefined' && contentRef.current
          ? new ResizeObserver(apply)
          : null
      observer?.observe(contentRef.current as Element)
      return () => {
        active = false
        window.removeEventListener('resize', apply)
        window.removeEventListener('scroll', apply, true)
        observer?.disconnect()
      }
    }, [open, place])

    /**
     * Re-measure after every commit that changed the anchor.
     *
     * The first placement runs while the menu is still laid out at `<body>`
     * width — it only becomes `position: fixed` once the anchor lands — so the
     * width that placement reads is not the width the menu will have, and `left`
     * clamps to the viewport floor instead of aligning to the trigger. Measuring
     * again from here sees the box as React actually rendered it, because this
     * effect runs after the style has been applied.
     *
     * Deliberately not rAF or a microtask: those run before the flush (or only
     * during rendering steps that never happen while the page is hidden), so they
     * re-measure the same stale rectangle. The equality check converges; the tick
     * cap guarantees it stops even if width and left keep chasing each other.
     */
    useLayoutEffect(() => {
      if (!open || !anchor) return
      if (corrections.current++ > 4) return
      const next = place()
      if (next && (next.top !== anchor.top || next.left !== anchor.left)) setAnchor(next)
    }, [open, anchor, place])

    useLayoutEffect(() => {
      if (open) {
        previousActiveElement.current = document.activeElement as HTMLElement
        requestAnimationFrame(() => focusFirst())
      } else if (previousActiveElement.current) {
        previousActiveElement.current.focus()
        previousActiveElement.current = null
      }
    }, [open, focusFirst])

    useEffect(() => {
      if (!open) return
      const handler = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          e.preventDefault()
          onOpenChange(false)
        } else if (e.key === 'ArrowDown') {
          e.preventDefault()
          focusNext(e.target as HTMLElement)
        } else if (e.key === 'ArrowUp') {
          e.preventDefault()
          focusPrev(e.target as HTMLElement)
        } else if (e.key === 'Home') {
          e.preventDefault()
          focusFirst()
        } else if (e.key === 'End') {
          e.preventDefault()
          const items = contentRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"]')
          items?.[items.length - 1]?.focus()
        }
      }
      document.addEventListener('keydown', handler)
      return () => document.removeEventListener('keydown', handler)
    }, [open, onOpenChange, focusFirst, focusNext, focusPrev])

    useEffect(() => {
      if (!open) return
      const handler = (e: MouseEvent) => {
        const target = e.target as Node
        if (!contentRef.current?.contains(target) && !triggerRef.current?.contains(target)) {
          onOpenChange(false)
        }
      }
      document.addEventListener('mousedown', handler)
      return () => document.removeEventListener('mousedown', handler)
    }, [open, onOpenChange, triggerRef])

    if (!open) return null

    return createPortal(
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menu"
        className={cn(
          'z-50 min-w-[10rem] max-h-[70vh] overflow-y-auto overscroll-contain',
          'bg-popover border border-border rounded-lg shadow-xl p-1',
          // No `duration-*` here: it sets transition-duration with the default
          // transition-property of `all`, which would animate the left/top this
          // component writes during placement. `.animate-in` owns the timing.
          'animate-in fade-in-0 zoom-in-95',
          className,
        )}
        {...props}
        style={{ ...style, ...(anchor ?? {}) }}
      >
        {children}
      </div>,
      document.body,
    )
  },
)
DropdownMenuContent.displayName = 'DropdownMenuContent'

/* ── Item ───────────────────────────────────────────────────────── */

interface DropdownMenuItemProps extends Omit<HTMLAttributes<HTMLDivElement>, 'onSelect'> {
  destructive?: boolean
  inset?: boolean
  disabled?: boolean
  /** Fired on click (in addition to onClick). Mirrors Radix onSelect. */
  onSelect?: (event: React.MouseEvent<HTMLDivElement>) => void
}

const DropdownMenuItem = forwardRef<HTMLDivElement, DropdownMenuItemProps>(
  ({ className, destructive, inset, disabled, onClick, onSelect, ...props }, ref) => {
    const { registerItem, unregisterItem, onOpenChange } = useDropdownMenuContext()
    const itemRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
      registerItem(itemRef.current)
      return () => unregisterItem(itemRef.current)
    }, [registerItem, unregisterItem])

    return (
      <div
        ref={(node) => {
          ;(itemRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menuitem"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        data-disabled={disabled ? '' : undefined}
        onClick={(e) => {
          if (disabled) return
          onClick?.(e)
          onSelect?.(e)
          onOpenChange(false)
        }}
        className={cn(
          'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
          'focus:bg-primary/10',
          'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
          destructive ? 'text-destructive focus:text-destructive' : 'text-foreground',
          inset && 'pl-8',
          className,
        )}
        {...props}
      />
    )
  },
)
DropdownMenuItem.displayName = 'DropdownMenuItem'

/* ── Checkbox Item ──────────────────────────────────────────────── */

interface DropdownMenuCheckboxItemProps extends HTMLAttributes<HTMLDivElement> {
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
  disabled?: boolean
}

const DropdownMenuCheckboxItem = forwardRef<HTMLDivElement, DropdownMenuCheckboxItemProps>(
  ({ className, children, checked, onCheckedChange, disabled, onClick, ...props }, ref) => {
    const { registerItem, unregisterItem, onOpenChange } = useDropdownMenuContext()
    const itemRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
      registerItem(itemRef.current)
      return () => unregisterItem(itemRef.current)
    }, [registerItem, unregisterItem])

    return (
      <div
        ref={(node) => {
          ;(itemRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menuitemcheckbox"
        aria-checked={checked}
        tabIndex={disabled ? -1 : 0}
        data-disabled={disabled ? '' : undefined}
        onClick={(e) => {
          if (disabled) return
          onCheckedChange?.(!checked)
          onClick?.(e)
        }}
        className={cn(
          'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
          'focus:bg-primary/10',
          'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
          className,
        )}
        {...props}
      >
        {children}
        {checked && (
          <span className="ml-auto">
            <svg className="h-4 w-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </span>
        )}
      </div>
    )
  },
)
DropdownMenuCheckboxItem.displayName = 'DropdownMenuCheckboxItem'

/* ── Radio Group / Radio Item ───────────────────────────────────── */

interface DropdownMenuRadioGroupContextValue {
  value: string
  onValueChange: (value: string) => void
}

const DropdownMenuRadioGroupContext = createContext<DropdownMenuRadioGroupContextValue | null>(null)

interface DropdownMenuRadioGroupProps {
  value?: string
  onValueChange?: (value: string) => void
  children: ReactNode
}

function DropdownMenuRadioGroup({ value = '', onValueChange, children }: DropdownMenuRadioGroupProps) {
  return (
    <DropdownMenuRadioGroupContext.Provider value={{ value, onValueChange: onValueChange ?? (() => {}) }}>
      <div role="group">{children}</div>
    </DropdownMenuRadioGroupContext.Provider>
  )
}

interface DropdownMenuRadioItemProps extends HTMLAttributes<HTMLDivElement> {
  value: string
  disabled?: boolean
}

const DropdownMenuRadioItem = forwardRef<HTMLDivElement, DropdownMenuRadioItemProps>(
  ({ className, children, value: itemValue, disabled, onClick, ...props }, ref) => {
    const groupCtx = useContext(DropdownMenuRadioGroupContext)
    const { registerItem, unregisterItem, onOpenChange } = useDropdownMenuContext()
    const itemRef = useRef<HTMLDivElement>(null)
    const isSelected = groupCtx?.value === itemValue

    useEffect(() => {
      registerItem(itemRef.current)
      return () => unregisterItem(itemRef.current)
    }, [registerItem, unregisterItem])

    return (
      <div
        ref={(node) => {
          ;(itemRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menuitemradio"
        aria-checked={isSelected}
        tabIndex={disabled ? -1 : 0}
        data-disabled={disabled ? '' : undefined}
        onClick={(e) => {
          if (disabled) return
          groupCtx?.onValueChange(itemValue)
          onClick?.(e)
          onOpenChange(false)
        }}
        className={cn(
          'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
          'focus:bg-primary/10',
          'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
          className,
        )}
        {...props}
      >
        {children}
        {isSelected && (
          <span className="ml-auto">
            <svg className="h-3 w-3 text-primary" fill="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="6" />
            </svg>
          </span>
        )}
      </div>
    )
  },
)
DropdownMenuRadioItem.displayName = 'DropdownMenuRadioItem'

/* ── Label ──────────────────────────────────────────────────────── */

interface DropdownMenuLabelProps extends HTMLAttributes<HTMLDivElement> {
  inset?: boolean
}

const DropdownMenuLabel = forwardRef<HTMLDivElement, DropdownMenuLabelProps>(
  ({ className, inset, ...props }, ref) => (
    <div ref={ref} className={cn('px-2.5 py-2 text-xs font-semibold text-muted-foreground', inset && 'pl-8', className)} {...props} />
  ),
)
DropdownMenuLabel.displayName = 'DropdownMenuLabel'

/* ── Separator ──────────────────────────────────────────────────── */

const DropdownMenuSeparator = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} role="separator" className={cn('my-1 h-px bg-border', className)} {...props} />,
)
DropdownMenuSeparator.displayName = 'DropdownMenuSeparator'

/* ── Sub Menu ───────────────────────────────────────────────────── */

interface DropdownMenuSubContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const DropdownMenuSubContext = createContext<DropdownMenuSubContextValue | null>(null)

interface DropdownMenuSubProps {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
}

function DropdownMenuSub({ open: controlledOpen, defaultOpen = false, onOpenChange, children }: DropdownMenuSubProps) {
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

  return <DropdownMenuSubContext.Provider value={{ open, onOpenChange: setOpen }}>{children}</DropdownMenuSubContext.Provider>
}

interface DropdownMenuSubTriggerProps extends HTMLAttributes<HTMLDivElement> {
  inset?: boolean
}

const DropdownMenuSubTrigger = forwardRef<HTMLDivElement, DropdownMenuSubTriggerProps>(
  ({ className, inset, children, ...props }, ref) => {
    const subCtx = useContext(DropdownMenuSubContext)
    const { registerItem, unregisterItem } = useDropdownMenuContext()
    const itemRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
      registerItem(itemRef.current)
      return () => unregisterItem(itemRef.current)
    }, [registerItem, unregisterItem])

    return (
      <div
        ref={(node) => {
          ;(itemRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menuitem"
        tabIndex={0}
        aria-haspopup="menu"
        aria-expanded={subCtx?.open}
        onMouseEnter={() => subCtx?.onOpenChange(true)}
        onMouseLeave={() => subCtx?.onOpenChange(false)}
        className={cn(
          'relative flex items-center gap-2 px-2.5 py-2 text-sm rounded-md cursor-pointer outline-none transition-colors duration-150',
          'focus:bg-primary/10 data-[state=open]:bg-primary/10',
          inset && 'pl-8',
          className,
        )}
        {...props}
      >
        {children}
        <svg className="ml-auto h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    )
  },
)
DropdownMenuSubTrigger.displayName = 'DropdownMenuSubTrigger'

const DropdownMenuSubContent = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => {
    const subCtx = useContext(DropdownMenuSubContext)
    const contentRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
      if (!subCtx?.open) return
      const handler = (e: MouseEvent) => {
        const target = e.target as Node
        if (!contentRef.current?.contains(target)) {
          subCtx.onOpenChange(false)
        }
      }
      document.addEventListener('mousedown', handler)
      return () => document.removeEventListener('mousedown', handler)
    }, [subCtx])

    if (!subCtx?.open) return null

    return createPortal(
      <div
        ref={(node) => {
          ;(contentRef as React.MutableRefObject<HTMLDivElement | null>).current = node
          if (typeof ref === 'function') ref(node)
          else if (ref) ref.current = node
        }}
        role="menu"
        className={cn(
          'z-50 min-w-[10rem] bg-popover border border-border rounded-lg shadow-xl p-1',
          'animate-in fade-in-0 zoom-in-95 duration-150',
          className,
        )}
        onMouseEnter={() => subCtx.onOpenChange(true)}
        onMouseLeave={() => subCtx.onOpenChange(false)}
        {...props}
      >
        {children}
      </div>,
      document.body,
    )
  },
)
DropdownMenuSubContent.displayName = 'DropdownMenuSubContent'

/* ── Group ──────────────────────────────────────────────────────── */

function DropdownMenuGroup({ children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div role="group" {...props}>
      {children}
    </div>
  )
}

/* ── Exports ────────────────────────────────────────────────────── */

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
  DropdownMenuGroup,
}
