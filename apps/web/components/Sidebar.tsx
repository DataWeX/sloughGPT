'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn, Button } from '@sloughgpt/strui'
import {
  IconChat,
  IconClose,
  IconModels,
  IconSettings,
  IconTraining,
  IconBrain,
  IconAgents,
  IconVision,
  IconSearch,
  IconCompare,
  IconActivity,
  IconLabs,
  IconDownload,
  IconAlert,
  IconBenchmark,
} from '@/components/icons/NavIcons'
import { routeMatchesPath } from '@/lib/route-match'
import { ThemeSwitcher } from './ThemeSwitcher'
import { useLocale } from '@/hooks/useLocale'

interface NavItem {
  path: string
  key: string
  Icon: typeof IconChat
}

interface NavSection {
  items: NavItem[]
}

const sections: NavSection[] = [
  {
    items: [
      { path: '/chat', key: 'nav.chat', Icon: IconChat },
      { path: '/training', key: 'nav.training', Icon: IconTraining },
      { path: '/datasets', key: 'nav.datasets', Icon: IconBrain },
      { path: '/knowledge', key: 'nav.knowledge', Icon: IconSearch },
      { path: '/companion', key: 'nav.companion', Icon: IconBrain },
    ],
  },
  {
    items: [
      { path: '/models', key: 'nav.models', Icon: IconModels },
      { path: '/agents', key: 'nav.agents', Icon: IconAgents },
      { path: '/multimodal', key: 'nav.multimodal', Icon: IconVision },
      { path: '/compare', key: 'nav.compare', Icon: IconCompare },
      { path: '/adapters', key: 'nav.adapters', Icon: IconSettings },
      { path: '/feedback', key: 'nav.feedback', Icon: IconActivity },
      { path: '/souls', key: 'nav.souls', Icon: IconBrain },
      { path: '/vm', key: 'nav.vm', Icon: IconLabs },
      { path: '/monitoring', key: 'nav.monitoring', Icon: IconActivity },
      { path: '/errors', key: 'nav.errors', Icon: IconAlert },
      { path: '/experiments', key: 'nav.experiments', Icon: IconBenchmark },
      { path: '/workflow', key: 'nav.workflow', Icon: IconActivity },
      { path: '/tokenizer', key: 'nav.tokenizer', Icon: IconSettings },
      { path: '/learn', key: 'nav.learn', Icon: IconSearch },
      { path: '/benchmark', key: 'nav.benchmark', Icon: IconBenchmark },
      { path: '/voice', key: 'nav.voice', Icon: IconActivity },
      { path: '/files', key: 'nav.files', Icon: IconSettings },
      { path: '/registry', key: 'nav.registry', Icon: IconModels },
      { path: '/security', key: 'nav.security', Icon: IconAlert },
      { path: '/images', key: 'nav.images', Icon: IconVision },
      { path: '/auth', key: 'nav.auth', Icon: IconSettings },
      { path: '/export', key: 'nav.export', Icon: IconDownload },
      { path: '/settings', key: 'nav.settings', Icon: IconSettings },
    ],
  },
]

const NAV_ICON = 'h-4 w-4 shrink-0'

export type SidebarVariant = 'desktop' | 'drawer'

export type SidebarProps = {
  variant?: SidebarVariant
  collapsed?: boolean
  onToggleCollapse?: () => void
  onNavigate?: () => void
  onClose?: () => void
}

export function Sidebar({ variant = 'desktop', collapsed = false, onToggleCollapse, onNavigate, onClose }: SidebarProps) {
  const pathname = usePathname()
  const isDrawer = variant === 'drawer'
  const isCollapsed = collapsed && !isDrawer
  const { t } = useLocale()

  const navLinkClass = (active: boolean) =>
    cn(
      'group relative flex min-h-11 items-center gap-3 rounded-lg py-2 text-sm transition-colors duration-200 ease-smooth',
      isCollapsed ? 'justify-center px-2' : 'px-3',
      active
        ? 'bg-primary/[0.13] font-medium text-primary dark:bg-primary/[0.11]'
        : 'text-foreground/78 hover:bg-primary/10 hover:text-primary dark:text-muted-foreground',
    )

  const afterNav = onNavigate ? () => onNavigate() : undefined

  return (
    <div className="relative flex h-dvh min-w-0 flex-1">
      <aside
        className={cn(
          'sl-sidebar-surface flex flex-col w-full min-w-0 overflow-hidden',
          isDrawer
            ? 'h-full pb-[max(0px,env(safe-area-inset-bottom))]'
            : 'h-dvh',
        )}
        data-collapsed={isCollapsed ? 'true' : undefined}
        aria-label="Main navigation"
      >
        <div
          className={cn(
            'flex h-[var(--app-mobile-header-h)] shrink-0 items-center border-b border-border/30 dark:border-border/50',
            isDrawer ? 'gap-1 px-2' : isCollapsed ? 'w-fit justify-center px-2' : 'px-3',
          )}
        >
          {isDrawer ? (
            <Button variant="menu" size="icon" aria-label={t('sidebar.close')} onClick={onClose}>
              <IconClose className="h-5 w-5" aria-hidden />
            </Button>
          ) : null}
          <Link
            href="/"
            className="flex min-w-0 flex-1 items-center gap-3 outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={t('sidebar.home')}
            onClick={afterNav}
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 font-mono text-sm font-bold leading-none tracking-tight text-primary-foreground shadow-sm">
              S
            </div>
            {!isCollapsed && (
              <div className="flex min-w-0 flex-col justify-center gap-0.5 leading-none">
                <span className="truncate text-sm font-semibold tracking-tight text-foreground">{t('app.name')}</span>
                {!isDrawer && (
                  <span className="text-[0.625rem] uppercase leading-tight tracking-wider text-muted-foreground">
                    {t('app.console')}
                  </span>
                )}
              </div>
            )}
          </Link>
        </div>

        <nav
          className={cn(
            'flex min-h-0 flex-1 flex-col',
            isCollapsed ? 'w-fit p-2' : 'p-3',
          )}
          aria-label="Primary"
        >
          <div className={cn('min-h-0 flex-1 overscroll-contain overflow-y-auto scrollbar-hide', isCollapsed && 'w-fit')}>
            {sections.map((section, si) => (
              <div key={si}>
                {si > 0 && (
                  <div className="my-2 border-t border-border/30 dark:border-border/40" />
                )}
                <ul className="space-y-0.5">
                  {section.items.map((item) => {
                    const active = routeMatchesPath(pathname, item.path)
                    return (
                      <li key={item.path}>
                        <Link
                          href={item.path}
                          aria-current={active ? 'page' : undefined}
                          className={navLinkClass(active)}
                          onClick={afterNav}
                          title={isCollapsed ? t(item.key) : undefined}
                        >
                          <item.Icon
                            className={cn(NAV_ICON, active ? 'opacity-100' : 'opacity-90 dark:opacity-80')}
                            aria-hidden
                          />
                          {!isCollapsed && (
                            <span className="flex-1 truncate">{t(item.key)}</span>
                          )}
                        </Link>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </div>
        </nav>

        {!isDrawer && (
          <div className={cn('shrink-0 border-t border-border/50 px-3 py-3', isCollapsed && 'w-fit')}>
            <ThemeSwitcher />
          </div>
        )}
      </aside>
    </div>
  )
}
