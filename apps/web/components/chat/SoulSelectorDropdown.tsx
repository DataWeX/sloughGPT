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

        <div className="py-1 max-h-48 overflow-y-auto">
          {souls.map((s) => (
            <DropdownMenuItem
              key={s.name}
              onSelect={() => onSelect(s)}
              className="justify-between text-xs px-3 py-1.5"
            >
              <div className="flex items-center gap-2">
                <IconHeart className="h-3 w-3 text-muted-foreground shrink-0" />
                <span className="capitalize">{s.name}</span>
              </div>
              {currentSoul?.name === s.name && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
            </DropdownMenuItem>
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
