'use client'

import { cn } from '@/lib/cn'

interface SectionHeaderProps {
  title: string
  className?: string
}

export function SectionHeader({ title, className }: SectionHeaderProps) {
  return (
    <p className={cn("text-[10px] font-medium text-muted-foreground uppercase tracking-wider px-1", className)}>
      {title}
    </p>
  )
}

interface SectionListProps {
  children: React.ReactNode
  className?: string
}

export function SectionList({ children, className }: SectionListProps) {
  return (
    <div className={cn("space-y-1 px-1", className)}>
      {children}
    </div>
  )
}

interface SectionBoxProps {
  children: React.ReactNode
  className?: string
}

export function SectionBox({ children, className }: SectionBoxProps) {
  return (
    <div className={cn("flex-1 mx-2 my-2 rounded-lg border border-border/50 bg-card/30 overflow-hidden", className)}>
      {children}
    </div>
  )
}

interface SectionScrollProps {
  children: React.ReactNode
  className?: string
}

export function SectionScroll({ children, className }: SectionScrollProps) {
  return (
    <div className={cn("h-full overflow-y-auto", className)}>
      {children}
    </div>
  )
}