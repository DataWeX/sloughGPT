'use client'

interface SearchEmptyStateProps {
  query?: string
  message?: string
}

export function SearchEmptyState({ query, message }: SearchEmptyStateProps) {
  const text = message ?? (query ? `No results for "${query}"` : 'Type to search across all workspace data')

  return (
    <p className="text-sm text-muted-foreground text-center py-8">{text}</p>
  )
}
