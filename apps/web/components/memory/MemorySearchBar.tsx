'use client'

import { Card, CardContent, Button, Input } from '@sloughgpt/strui'

interface MemorySearchBarProps {
  query: string
  onQueryChange: (q: string) => void
  onSearch: () => void
  onClear: () => void
  searching?: boolean
  hasResults?: boolean
}

export function MemorySearchBar({ query, onQueryChange, onSearch, onClear, searching, hasResults }: MemorySearchBarProps) {
  return (
    <Card>
      <CardContent className="p-2.5">
        <div className="flex items-center gap-1.5">
          <Input
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') onSearch() }}
            placeholder="Search memory..."
            className="h-7 text-[11px] flex-1"
          />
          <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={onSearch} disabled={searching}>
            {searching ? 'Searching...' : 'Search'}
          </Button>
          {hasResults && (
            <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={onClear}>
              Clear
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
