'use client'

import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/cn'

interface MenuItem {
  label?: string
  icon?: React.ReactNode
  onClick?: () => void
  destructive?: boolean
  separator?: boolean
  custom?: React.ReactNode
}

interface CustomDropdownProps {
  trigger: React.ReactNode
  items: MenuItem[]
  align?: 'start' | 'end'
  grid?: boolean
  className?: string
}

export function CustomDropdown({ trigger, items, align = 'start', grid = false, className }: CustomDropdownProps) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  return (
    <div className="relative">
      <div ref={triggerRef} onClick={() => setOpen(!open)}>
        {trigger}
      </div>

      {open && (
        <div
          ref={menuRef}
          className={cn(
            'absolute z-50 left-0 right-0 w-full p-1 rounded-md border border-border bg-popover shadow-lg animate-in fade-in zoom-in-95 duration-100 overflow-hidden',
            'bottom-full mb-0.5',
            grid && 'grid grid-cols-2 gap-1',
            className
          )}
        >
          {items.map((item, i) =>
            item.separator ? (
              <div key={i} className="h-px bg-border/50 col-span-2 w-full" />
            ) : item.custom ? (
              <div key={i} className="col-span-2 w-full">{item.custom}</div>
            ) : (
              <button
                key={i}
                onClick={() => {
                  item.onClick?.()
                  setOpen(false)
                }}
                className={cn(
                  'w-full',
                   grid
                    ? 'flex flex-col items-center justify-center gap-0.5 p-1.5 text-xs rounded-sm hover:bg-primary/10'
                    : 'flex items-center gap-2 px-2 py-2 text-sm rounded-sm hover:bg-primary/10',
                  item.destructive
                    ? 'text-destructive hover:bg-destructive/10'
                    : ''
                )}
              >
                {item.icon && <span className={grid ? 'h-4 w-4' : 'h-4 w-4 shrink-0'}>{item.icon}</span>}
                <span className="truncate text-center">{item.label}</span>
              </button>
            )
          )}
        </div>
      )}
    </div>
  )
}
