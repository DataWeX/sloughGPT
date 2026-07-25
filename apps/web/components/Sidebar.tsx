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
    ],
  },
  {
    items: [
      { path: '/models', key: 'nav.models', Icon: IconModels },
      { path: '/agents', key: 'nav.agents', Icon: IconAgents },
      { path: '/multimodal', key: 'nav.multimodal', Icon: IconVision },
      { path: '/settings', key: 'nav.settings', Icon: IconSettings },
    ],
  },
]

const NAV_ICON = 'h-[1.125rem] w-[1.125rem] shrink-0'

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
      'group relative flex min-h-[2.5rem] items-center gap-3 rounded-lg py-2 text-sm transition-colors duration-200 ease-smooth',
      isCollapsed ? 'justify-center px-2' : 'px-3',
      active
        ? 'bg-primary/[0.13] font-medium text-primary dark:bg-primary/[0.11]'
        : 'text-foreground/78 hover:bg-primary/10 hover:text-primary dark:text-muted-foreground',
    )

  const afterNav = onNavigate ? () => onNavigate() : undefined

  return (
    <div className="relative flex h-dvh shrink-0 overflow-visible">
      <aside
        className={cn(
          'sl-sidebar-surface flex shrink-0 flex-col',
          isDrawer
            ? 'h-full w-full min-w-0 pb-[max(0px,env(safe-area-inset-bottom))]'
            : 'h-dvh',
        )}
        data-collapsed={isCollapsed ? 'true' : undefined}
        aria-label="Main navigation"
      >
        <div
          className={cn(
            'flex h-[3.25rem] shrink-0 items-center border-b border-border/30 dark:border-border/50',
            isDrawer ? 'gap-1 px-2' : 'px-3',
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
            isCollapsed ? 'p-2' : 'p-3',
          )}
          aria-label="Primary"
        >
          <div className="min-h-0 flex-1 overscroll-contain overflow-y-auto scrollbar-hide">
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
          <div className="shrink-0 border-t border-border/50 px-3 py-3">
            <ThemeSwitcher />
          </div>
        )}
      </aside>

      {/* 3D bookmark tab — protrudes from sidebar edge */}
      {!isDrawer && onToggleCollapse && (
        <button
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className={cn(
            'group/tab absolute top-1/2 z-20 flex h-[4.5rem] w-[1.15rem] -translate-y-1/2 items-center justify-center',
            'rounded-r-md cursor-pointer',
            'transition-all duration-300 ease-[cubic-bezier(222,133,0,1)]',
            'hover:h-[5rem]',
            /* Sit on the right edge, protruding outside the sidebar border */
            'right-0 translate-x-[calc(100%-1px)]',
            /* 3D gradient background */
            'bg-gradient-to-b from-primary/80 via-primary to-primary/90',
            /* 3D depth — layered shadows */
            'shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1)]',
            /* Inner highlight for top edge 3D feel */
            'before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px before:rounded-r-md before:bg-gradient-to-b before:from-white/40 before:to-transparent',
            /* Subtle inner shadow for bottom edge */
            'after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-px after:rounded-r-md after:bg-gradient-to-t after:from-black/20 after:to-transparent',
            /* Hover glow */
            'hover:shadow-[1px_0_2px_-1px_rgba(0,0,0,0.2),2px_0_4px_-2px_rgba(0,0,0,0.15),3px_0_8px_-3px_rgba(0,0,0,0.1),0_0_12px_-2px_rgba(var(--primary)/0.3)]',
          )}
        >
          {/* Arrow icon */}
          <svg
            className={cn(
              'h-3 w-3 text-primary-foreground transition-transform duration-300',
              'drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]',
              isCollapsed ? 'rotate-0' : 'rotate-180',
              'group-hover/tab:scale-110',
            )}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M9 18l6-6-6-6" />
          </svg>
        </button>
      )}
    </div>
  )
}
