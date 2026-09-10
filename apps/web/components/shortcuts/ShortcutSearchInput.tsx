'use client'

import { Input } from '@sloughgpt/strui'
import { Search } from 'lucide-react'

interface ShortcutSearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
}

export function ShortcutSearchInput({ value, onChange, placeholder = 'Search shortcuts...' }: ShortcutSearchInputProps) {
  return (
    <div className="relative">
      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
      <Input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="h-7 text-[11px] pl-7"
        aria-label="Search shortcuts"
      />
    </div>
  )
}
