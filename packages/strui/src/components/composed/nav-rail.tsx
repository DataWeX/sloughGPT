import type { ComponentPropsWithoutRef, ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface NavRailProps {
  children: ReactNode
  className?: string
  /** Visually group the first item(s) as a product / home link. */
  header?: ReactNode
}

/** Vertical navigation strip using the lattice sidebar surface token. */
export function NavRail({ children, className, header }: NavRailProps) {
  return (
    <nav
      className={cn(
        'sl-sidebar-surface flex h-full min-h-0 w-[min(100%,var(--sidebar-width))] flex-col gap-0.5 p-2 sm:p-3',
        className,
      )}
      aria-label="Main"
    >
      {header ? <div className="mb-2 border-b border-border/60 pb-2">{header}</div> : null}
      <div className="flex flex-col gap-0.5">{children}</div>
    </nav>
  )
}

export type NavRailLinkProps = ComponentPropsWithoutRef<'a'> & {
  active?: boolean
}

export function NavRailLink({ className, active, children, ...props }: NavRailLinkProps) {
  return (
    <a
      className={cn(
        'str-touch-target flex items-center gap-2 rounded-none border border-transparent px-3 py-2.5 text-sm font-medium transition-colors duration-200 ease-smooth',
        'sl-hover-primary',
        active && 'border-border bg-muted/60 text-foreground',
        className,
      )}
      {...props}
    >
      {children}
    </a>
  )
}
