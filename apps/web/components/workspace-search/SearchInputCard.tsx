'use client'

import { Card, CardContent, Input } from '@sloughgpt/strui'
import { Search } from 'lucide-react'

interface SearchInputCardProps {
  query: string
  onQueryChange: (query: string) => void
  placeholder?: string
}

export function SearchInputCard({
  query,
  onQueryChange,
  placeholder = 'Search members, training jobs, datasets, knowledge...',
}: SearchInputCardProps) {
  return (
    <Card className="mb-6">
      <CardContent className="py-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder={placeholder}
            className="pl-9"
            autoFocus
          />
        </div>
      </CardContent>
    </Card>
  )
}
