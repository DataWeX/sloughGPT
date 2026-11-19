'use client'

import Link from '@/vite/next-compat/link'
import { usePathname } from '@/vite/next-compat/navigation'
import { cn } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { ALL_ROUTES, SIDEBAR_ICONS } from '@/lib/navigation'

/** Curated mobile subset — membership is the only thing this file decides;
 *  labelKeys and icons project from navigation.ts (the single source). */
const BOTTOM_NAV_PATHS = ['/chat', '/training', '/knowledge', '/settings'] as const

const BOTTOM_NAV_ITEMS = BOTTOM_NAV_PATHS.map((path) => {
  const route = ALL_ROUTES.find((r) => r.path === path)
  if (!route) throw new Error(`BottomNav: ${path} missing from NAV_SECTIONS`)
  const icon = SIDEBAR_ICONS[path]
  if (!icon) throw new Error(`BottomNav: ${path} missing from SIDEBAR_ICONS`)
  return { path, labelKey: route.labelKey, icon }
})

export function BottomNav() {
  const pathname = usePathname()
  const { t } = useLocale()

  const isActive = (path: string) => {
    if (path === '/chat') return pathname === '/' || pathname === '/chat'
    return pathname.startsWith(path)
  }

  return (
    <nav className="sl-bottom-nav" aria-label="Bottom navigation">
      {BOTTOM_NAV_ITEMS.map((item) => {
        const active = isActive(item.path)
        const Icon = item.icon
        return (
          <Link
            key={item.path}
            href={item.path}
            aria-label={t(item.labelKey)}
            aria-current={active ? 'page' : undefined}
            className={cn('sl-bottom-nav-item', active && 'sl-bottom-nav-item--active')}
          >
            <Icon className="h-5 w-5" />
            <span className="sl-bottom-nav-label">{t(item.labelKey)}</span>
          </Link>
        )
      })}
    </nav>
  )
}
