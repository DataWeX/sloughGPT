'use client'

import { cn } from '@sloughgpt/strui'
import { useAuthStore } from '@/lib/auth'
import { IconChevronDown, IconBuilding } from '@/components/icons/NavIcons'

export function WorkspaceSwitcher() {
  const { workspaces, currentWorkspace, switchWorkspace } = useAuthStore()

  if (workspaces.length <= 1) return null

  return (
    <div className="relative group">
      <button
        type="button"
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm',
          'bg-muted/50 hover:bg-muted transition-colors',
          'border border-border/60 hover:border-border'
        )}
      >
        <IconBuilding className="h-4 w-4 text-muted-foreground" />
        <span className="truncate max-w-[120px]">
          {currentWorkspace?.name ?? 'Select workspace'}
        </span>
        <IconChevronDown className="h-3 w-3 text-muted-foreground" />
      </button>

      <div className={cn(
        'absolute top-full left-0 mt-1 z-50',
        'bg-card border border-border/60 rounded-lg shadow-lg',
        'opacity-0 invisible group-hover:opacity-100 group-hover:visible',
        'transition-all duration-150',
        'min-w-[200px] p-1'
      )}>
        {workspaces.map(ws => (
          <button
            key={ws.id}
            type="button"
            onClick={() => switchWorkspace(ws.id)}
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm text-left',
              'hover:bg-muted/50 transition-colors',
              ws.id === currentWorkspace?.id && 'bg-primary/[0.08] text-primary'
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
