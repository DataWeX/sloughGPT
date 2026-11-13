'use client'

import type { ReactNode } from 'react'
import Link from '@/vite/next-compat/link'
import { usePathname } from '@/vite/next-compat/navigation'
import { cn } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { routeMatchesPath } from '@/lib/route-match'

export interface WorkspaceTab {
  path: string
  labelKey: string
}

/** Safe pathname — unit tests may render without a Router. */
function useSafePathname(): string {
  try {
    return usePathname()
  } catch {
    return typeof window !== 'undefined' ? window.location.pathname : '/'
  }
}

/** Safe locale — unit tests may render without LocaleProvider. */
function useSafeLocale(): { t: (key: string) => string } {
  try {
    return useLocale()
  } catch {
    return { t: (key: string) => key }
  }
}

const tabClass = (active: boolean) =>
  cn(
    'inline-flex min-h-10 items-center border-b-2 px-3 text-[11px] transition-colors duration-200 ease-smooth outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring',
    active
      ? 'border-primary font-medium text-primary'
      : 'border-transparent text-muted-foreground hover:border-border hover:text-foreground',
  )

/** Pathname-driven tab bar for workspace section layouts. Deep-linkable. */
export function WorkspaceSectionTabs({ tabs, ariaLabel }: { tabs: WorkspaceTab[]; ariaLabel: string }) {
  const pathname = useSafePathname()
  const { t } = useSafeLocale()

  let bestPath: string | null = null
  let bestLen = -1
  for (const tab of tabs) {
    if (routeMatchesPath(pathname, tab.path) && tab.path.length > bestLen) {
      bestPath = tab.path
      bestLen = tab.path.length
    }
  }

  return (
    <nav
      aria-label={ariaLabel}
      className="mb-4 flex flex-wrap gap-1 border-b border-border/40 dark:border-border/50"
    >
      {tabs.map((tab) => {
        const active = tab.path === bestPath
        return (
          <Link
            key={tab.path}
            href={tab.path}
            prefetch={false}
            aria-current={active ? 'page' : undefined}
            className={tabClass(active)}
          >
            {t(tab.labelKey)}
          </Link>
        )
      })}
    </nav>
  )
}

export function WorkspaceSectionLayout({
  tabs,
  ariaLabel,
  children,
}: {
  tabs: WorkspaceTab[]
  ariaLabel: string
  children: ReactNode
}) {
  return (
    <div className="min-w-0">
      <WorkspaceSectionTabs tabs={tabs} ariaLabel={ariaLabel} />
      {children}
    </div>
  )
}
