'use client'

import { useState } from 'react'
import { Input } from '@sloughgpt/strui'
import { Search } from 'lucide-react'

export interface ShortcutSearchFilterProps {
  categories: string[]
  onFilter: (filtered: string[]) => void
}

export function ShortcutSearchFilter({ categories, onFilter }: ShortcutSearchFilterProps) {
  const [query, setQuery] = useState('')

  const handleChange = (value: string) => {
    setQuery(value)
    if (!value.trim()) {
      onFilter(categories)
    } else {
      onFilter(categories.filter(c => c.toLowerCase().includes(value.toLowerCase())))
    }
  }

  return (
    <div className="flex items-center gap-2">
      <Search className="h-3.5 w-3.5 text-muted-foreground" />
      <Input
        value={query}
        onChange={e => handleChange(e.target.value)}
        placeholder="Filter categories..."
        className="h-7 text-[11px] flex-1"
        aria-label="Filter shortcuts by category"
      />
    </div>
  )
}
