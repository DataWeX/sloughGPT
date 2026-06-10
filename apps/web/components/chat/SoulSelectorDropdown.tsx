'use client'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { IconChevronDown, IconCheck, IconHeart } from '@/components/ui'
import type { Soul } from '@/lib/souls-controller'

interface SoulSelectorDropdownProps {
  souls: Soul[]
  currentSoul: Soul | null
  onSelect: (soul: Soul) => void
}

export function SoulSelectorDropdown({ souls, currentSoul, onSelect }: SoulSelectorDropdownProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-1.5 sm:px-2.5 text-xs gap-1 sm:gap-1.5 rounded-full bg-primary/8 text-primary hover:bg-primary/15 border border-primary/15" title={currentSoul?.traits?.join(', ')}>
          <IconHeart className="h-3 w-3 shrink-0" />
          <span className="hidden sm:inline truncate max-w-[64px]">{currentSoul?.name || 'Personality'}</span>
          <IconChevronDown className="hidden sm:block h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[160px]">
        {souls.map((s) => (
          <DropdownMenuItem
            key={s.name}
            onSelect={() => onSelect(s)}
            className="justify-between text-xs"
          >
            <span>{s.name}</span>
            {currentSoul?.name === s.name && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
