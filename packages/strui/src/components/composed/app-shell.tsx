import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface AppShellProps {
  /** Desktop sidebar; hidden below `md` unless `showSidebarMobile` is true. */
  sidebar?: ReactNode
  /** Sticky top bar (e.g. mobile header + token meter). */
  topBar?: ReactNode
  children: ReactNode
  className?: string
  /** When set, sidebar is shown in a column above main on small screens (narrow layouts). */
  showSidebarMobile?: boolean
}

/**
 * Full-viewport shell: optional sidebar + main. Uses `str-min-h-screen` and safe-area–friendly structure.
 * Pair **NavRail** in `sidebar` and **PageHeader** inside `children` as needed.
 */
export function AppShell({ sidebar, topBar, children, className, showSidebarMobile }: AppShellProps) {
  return (
    <div className={cn('str-min-h-screen flex flex-col bg-background md:flex-row', className)}>
      {sidebar ? (
        <div
          className={cn(
            'shrink-0 border-border md:border-r',
            showSidebarMobile ? 'block' : 'hidden md:block',
          )}
        >
          {sidebar}
        </div>
      ) : null}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {topBar}
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </div>
    </div>
  )
}
