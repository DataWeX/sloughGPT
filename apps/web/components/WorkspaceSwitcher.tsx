'use client'

import { cn } from '@sloughgpt/strui'
import { useAuthStore } from '@/lib/auth'
import { IconChevronDown, IconBuilding } from '@/components/icons/NavIcons'

type WorkspaceSwitcherProps = {
  /**
   * Sidebar collapsed — the aside is overflow-hidden and only 3rem wide, so a
   * dropdown has no room to open. Render the icon-only trigger instead; a
   * click expands the nav (onExpand) so the full switcher can open.
   */
  collapsed?: boolean
  onExpand?: () => void
}

const TRIGGER_CLASS = cn(
  'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm',
  'bg-muted/50 hover:bg-muted transition-colors',
  'border border-border/60 hover:border-border',
)

export function WorkspaceSwitcher({ collapsed = false, onExpand }: WorkspaceSwitcherProps) {
  const { workspaces, currentWorkspace, switchWorkspace } = useAuthStore()

  const name = currentWorkspace?.name ?? 'Select workspace'

  // Collapsed rail: icon-only, click expands the sidebar (the menu itself
  // cannot open inside a 3rem overflow-hidden column).
  if (collapsed) {
    return (
      <button
        type="button"
        onClick={onExpand}
        title={`${name} — switch workspace`}
        aria-label={`${name} — switch workspace`}
        className={cn(TRIGGER_CLASS, 'w-8 justify-center px-0')}
      >
        <IconBuilding className="h-4 w-4 text-muted-foreground" />
      </button>
    )
  }

  // Single workspace: no menu to open, but still show which workspace this
  // session is on — the switcher must be visible, not silently absent.
  if (workspaces.length <= 1) {
    return (
      <div className={cn(TRIGGER_CLASS, 'cursor-default')}>
        <IconBuilding className="h-4 w-4 text-muted-foreground" />
        <span className="truncate max-w-[120px]">{name}</span>
      </div>
    )
  }

  return (
    <div className="relative group">
      <button type="button" className={TRIGGER_CLASS}>
        <IconBuilding className="h-4 w-4 text-muted-foreground" />
        <span className="truncate max-w-[120px]">{name}</span>
        <IconChevronDown className="h-3 w-3 text-muted-foreground" />
      </button>

      <div
        className={cn(
          'absolute top-full left-0 mt-1 z-50',
          'bg-card border border-border/60 rounded-lg shadow-lg',
          'opacity-0 invisible group-hover:opacity-100 group-hover:visible',
          'transition-all duration-150',
          'min-w-[200px] p-1',
        )}
      >
        {workspaces.map((ws) => (
          <button
            key={ws.id}
            type="button"
            onClick={() => switchWorkspace(ws.id)}
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm text-left',
              'hover:bg-muted/50 transition-colors',
              ws.id === currentWorkspace?.id && 'bg-primary/[0.08] text-primary',
            )}
          >
            <IconBuilding className="h-4 w-4 shrink-0" />
            <div className="min-w-0">
              <div className="truncate font-medium">{ws.name}</div>
              {ws.description && (
                <div className="truncate text-xs text-muted-foreground">{ws.description}</div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
