'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@sloughgpt/strui'
import { IconChat, IconTraining, IconBookmark, IconModels, IconSettings } from '@/components/icons/NavIcons'
import { useLocale } from '@/hooks/useLocale'
import type { ComponentType } from 'react'

interface BottomNavItem {
  path: string
  labelKey: string
  icon: ComponentType<{ className?: string }>
}

const BOTTOM_NAV_ITEMS: BottomNavItem[] = [
  { path: '/chat', labelKey: 'nav.chat', icon: IconChat },
  { path: '/training', labelKey: 'nav.training', icon: IconTraining },
  { path: '/knowledge', labelKey: 'nav.knowledge', icon: IconBookmark },
  { path: '/models', labelKey: 'nav.models', icon: IconModels },
  { path: '/settings', labelKey: 'nav.settings', icon: IconSettings },
]

export function BottomNav() {
  const pathname = usePathname()
  const { t } = useLocale()

  const isActive = (path: string) => {
    if (path === '/chat') return pathname === '/' || pathname === '/chat'
    return pathname.startsWith(path)
  }

  return (
    <nav className="sl-bottom-nav" aria-label="Bottom navigation">
      {BOTTOM_NAV_ITEMS.map(item => {
        const active = isActive(item.path)
        const Icon = item.icon
        return (
          <Link
            key={item.path}
            href={item.path}
            aria-label={t(item.labelKey)}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'sl-bottom-nav-item',
              active && 'sl-bottom-nav-item--active',
            )}
          >
            <Icon className="h-5 w-5" />
            <span className="sl-bottom-nav-label">{t(item.labelKey)}</span>
          </Link>
        )
      })}
    </nav>
  )
}
