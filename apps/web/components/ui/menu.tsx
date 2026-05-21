'use client'

import { type ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface TooltipProps {
  children: ReactNode
  content: string
  side?: 'top' | 'right' | 'bottom' | 'left'
  className?: string
}

export function Tooltip({ children, content, side = 'top', className }: TooltipProps) {
  return (
    <div className="relative group inline-block">
      {children}
      <div
        className={cn(
          "absolute z-50 px-2 py-1 text-xs rounded bg-foreground text-background opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity whitespace-nowrap pointer-events-none",
          side === 'top' && "bottom-full left-1/2 -translate-x-1/2 mb-1",
          side === 'bottom' && "top-full left-1/2 -translate-x-1/2 mt-1",
          side === 'left' && "right-full top-1/2 -translate-y-1/2 mr-1",
          side === 'right' && "left-full top-1/2 -translate-y-1/2 ml-1",
          className
        )}
      >
        {content}
      </div>
    </div>
  )
}

interface DropdownProps {
  trigger: ReactNode
  children: ReactNode
  align?: 'start' | 'end'
  className?: string
}

export function Dropdown({ trigger, children, align = 'start', className }: DropdownProps) {
  return (
    <div className="relative group inline-block">
      {trigger}
      <div
        className={cn(
          "absolute z-50 min-w-[140px] p-1 rounded-md border bg-background shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity",
          align === 'start' && "left-0",
          align === 'end' && "right-0",
          className
        )}
      >
        {children}
      </div>
    </div>
  )
}

interface MenuItemProps {
  children: ReactNode
  onClick?: () => void
  className?: string
}

export function MenuItem({ children, onClick, className }: MenuItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded hover:bg-primary/10 text-left",
        className
      )}
    >
      {children}
    </button>
  )
}

interface MenuDividerProps {
  className?: string
}

export function MenuDivider({ className }: MenuDividerProps) {
  return <div className={cn("h-px bg-border/50 my-1", className)} />
}

interface MenuProps {
  trigger: ReactNode
  children: ReactNode
  align?: 'start' | 'end'
  className?: string
}

export function Menu({ trigger, children, align = 'start', className }: MenuProps) {
  return (
    <div className="relative group">
      {trigger}
      <div
        className={cn(
          "absolute z-50 min-w-[160px] p-1 rounded-md border bg-background shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity",
          align === 'start' && "left-0",
          align === 'end' && "right-0",
          className
        )}
      >
        {children}
      </div>
    </div>
  )
}