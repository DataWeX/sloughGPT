'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconChevronDown, IconCheck, IconHeart, IconChevronRight } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

export function SoulSelectorDropdown() {
  const ctx = useChatToolbarContext()
  const { souls, current: currentSoul, onSelect } = ctx.soul
  const router = useRouter()
  const [open, setOpen] = useState(false)

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-1.5 sm:px-2.5 text-xs gap-1 sm:gap-1.5 rounded-full bg-primary/8 text-primary hover:bg-primary/15 border border-primary/15" title={currentSoul?.traits?.join(', ')}>
          <IconHeart className="h-3 w-3 shrink-0" />
          <span className="hidden sm:inline truncate max-w-[64px]">{currentSoul?.name || 'Personality'}</span>
          <IconChevronDown className="hidden sm:block h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52 p-0">
        <div className="px-3 pt-3 pb-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold capitalize">{currentSoul?.name || 'Default'}</span>
            {currentSoul?.traits && currentSoul.traits.length > 0 && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                {currentSoul.traits[0]}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => { setOpen(false); router.push('/models') }}
            className="inline-flex items-center gap-1 text-[10px] text-primary hover:text-primary/80 transition-colors mt-1"
          >
            View full profile
            <IconChevronRight className="h-2.5 w-2.5" />
          </button>
        </div>

        <div className="border-t border-border/40" />

        <div className="py-1 max-h-56 overflow-y-auto">
          {souls.map((s) => {
            const isActive = currentSoul?.name === s.name
            return (
              <DropdownMenuItem
                key={s.name}
                onSelect={() => onSelect(s)}
                className={`flex items-start gap-2 px-3 py-2 text-xs ${isActive ? 'bg-primary/8' : ''}`}
              >
                <IconHeart className={`h-3 w-3 mt-0.5 shrink-0 ${isActive ? 'text-primary' : 'text-muted-foreground'}`} />
                <div className="min-w-0 flex-1">
                  <div className="font-medium capitalize truncate">{s.name}</div>
                  {s.description && (
                    <div className="text-[10px] text-muted-foreground truncate mt-0.5">{s.description}</div>
                  )}
                  {s.traits && s.traits.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {s.traits.slice(0, 3).map((t) => (
                        <span key={t} className="text-[9px] px-1 py-0 rounded bg-muted text-muted-foreground">{t}</span>
                      ))}
                      {s.traits.length > 3 && (
                        <span className="text-[9px] text-muted-foreground">+{s.traits.length - 3}</span>
                      )}
                    </div>
                  )}
                </div>
                {isActive && <IconCheck className="h-3 w-3 text-primary shrink-0 mt-0.5" />}
              </DropdownMenuItem>
            )
          })}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
