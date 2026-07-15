'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Button } from '@sloughgpt/strui'
import {
  IconChat,
  IconClose,
  IconModels,
  IconSearch,
  IconSettings,
  IconActivity,
  IconCompare,
  IconTraining,
  IconExport,
  IconAlert,
  IconBrain,
} from '@/components/icons/NavIcons'
import { IconChevronDown } from '@sloughgpt/strui'
import { cn } from '@/lib/cn'
import { routeMatchesPath } from '@/lib/route-match'
import { ThemeSwitcher } from './ThemeSwitcher'
import { CustomDropdown } from './CustomDropdown'
import { useLocale, LOCALES } from '@/hooks/useLocale'

const allItems = [
  { path: '/chat', key: 'nav.chat', Icon: IconChat },
  { path: '/training', key: 'nav.training', Icon: IconTraining },
  { path: '/knowledge', key: 'nav.knowledge', Icon: IconSearch },
  { path: '/datasets', key: 'nav.datasets', Icon: IconBrain },
  { path: '/export', key: 'nav.export', Icon: IconExport },
  { path: '/compare', key: 'nav.compare', Icon: IconCompare },
  { path: '/monitoring', key: 'nav.monitoring', Icon: IconActivity },
  { path: '/models', key: 'nav.models', Icon: IconModels },
  { path: '/errors', key: 'nav.errors', Icon: IconAlert },
  { path: '/settings', key: 'nav.settings', Icon: IconSettings },
] as const

/** Sidebar list — smaller than home quick-action tiles */
const NAV_ICON = 'h-[1.125rem] w-[1.125rem] shrink-0'

export type SidebarVariant = 'desktop' | 'drawer'

export type SidebarProps = {
  variant?: SidebarVariant
  /** Called after choosing a nav link (e.g. close mobile drawer). */
  onNavigate?: () => void
  /** Close control for drawer variant. */
  onClose?: () => void
}

export function Sidebar({ variant = 'desktop', onNavigate, onClose }: SidebarProps) {
  const pathname = usePathname()
  const isDrawer = variant === 'drawer'
  const { t } = useLocale()

  const navLinkClass = (active: boolean) =>
    cn(
      'group relative flex min-h-[2.5rem] items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors duration-200 ease-smooth',
      active
        ? 'bg-primary/[0.13] font-medium text-primary dark:bg-primary/[0.11]'
: 'text-foreground/78 hover:bg-primary/10 hover:text-primary dark:text-muted-foreground',
  )

  const afterNav = onNavigate ? () => onNavigate() : undefined

  return (
    <aside
      className={cn(
        'sl-sidebar-surface flex shrink-0 flex-col',
        isDrawer
          ? 'h-full w-full min-w-0 pb-[max(0px,env(safe-area-inset-bottom))]'
          : 'h-dvh w-[var(--sidebar-width)]',
      )}
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
          <div className="flex min-w-0 flex-col justify-center gap-0.5 leading-none">
            <span className="truncate text-sm font-semibold tracking-tight text-foreground">{t('app.name')}</span>
            {!isDrawer && (
              <span className="text-[0.625rem] uppercase leading-tight tracking-wider text-muted-foreground">
                {t('app.console')}
              </span>
            )}
          </div>
        </Link>
      </div>

      <nav
        className="flex min-h-0 flex-1 flex-col p-3"
        aria-label="Primary"
      >
        <div className={cn("min-h-0 flex-1 overscroll-contain", isDrawer ? "overflow-y-auto scrollbar-hide" : "overflow-y-auto")}>
          <ul className="space-y-1">
            {allItems.map((item) => {
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
                    <span className="flex-1 truncate">{t(item.key)}</span>
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
      </nav>

      <div className="shrink-0 space-y-1 border-t border-border/50 px-3 py-3">
        <ThemeSwitcher />
        <CustomDropdown
          align="start"
          trigger={
            <button className="flex items-center justify-between gap-2 w-full px-3 py-2 hover:bg-accent/50 rounded-lg transition-colors cursor-pointer text-sm">
              <span className="h-7 w-7 shrink-0 rounded-full bg-primary text-primary-foreground text-sm font-medium flex items-center justify-center">
                A
              </span>
              <span className="flex-1 truncate text-sm text-left">{t('sidebar.account')}</span>
              <IconChevronDown className="h-4 w-4 shrink-0 opacity-50" aria-hidden />
            </button>
          }
          items={[
            {
              label: `${t('sidebar.signOut')} (offline)`,
              onClick: () => {
                import('@/lib/toast-store').then(({ useToastStore }) =>
                  useToastStore.getState().addToast('Auth is running in offline mode. Sign-in coming soon.')
                ).catch(() => {})
              },
            },
          ]}
        />
      </div>
    </aside>
  )
}
