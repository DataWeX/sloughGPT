'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'

interface Doc {
  _id: string
}

interface DocstoreBrowserCardProps {
  collections: string[]
  selected: string
  docs: Doc[]
  page: number
  totalPages: number
  loading: boolean
  searchQuery: string
  onSelect: (id: string) => void
  onSearchChange: (query: string) => void
  onPageChange: (page: number) => void
  onCreateToggle: () => void
  showCreate: boolean
}

export function DocstoreBrowserCard({
  collections,
  selected,
  docs,
  page,
  totalPages,
  loading,
  searchQuery,
  onSelect,
  onSearchChange,
  onPageChange,
  onCreateToggle,
  showCreate,
}: DocstoreBrowserCardProps) {
  return (
    <Card data-testid="docstore-browser">
      <CardHeader>
        <CardTitle className="text-base">Documents</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-1 flex-wrap">
          {collections.map(col => (
            <Button
              key={col}
              size="sm"
              variant={selected === col ? 'default' : 'ghost'}
              className="h-7 text-xs"
              data-testid={`collection-${col}`}
            >
              {col}
            </Button>
          ))}
        </div>

        <Input
          placeholder="Search documents..."
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
          data-testid="doc-search"
        />

        <div className="flex justify-between items-center">
          <Button
            size="sm"
            variant={showCreate ? 'secondary' : 'default'}
            onClick={onCreateToggle}
            data-testid="new-doc-btn"
          >
            {showCreate ? 'Cancel' : 'New doc'}
          </Button>
        </div>

        {loading ? (
          <div className="text-sm text-muted-foreground" data-testid="loading-indicator">Loading...</div>
        ) : docs.length === 0 ? (
          <div className="text-sm text-muted-foreground">No documents found.</div>
        ) : (
          <div className="space-y-1" data-testid="doc-list">
            {docs.map(doc => (
              <div
                key={doc._id}
                className={cn(
                  'flex items-center gap-2 rounded border p-2 text-sm cursor-pointer transition-colors',
                  'border-border/60 hover:bg-muted/50'
                )}
                data-testid={`doc-item-${doc._id}`}
                onClick={() => onSelect(doc._id)}
              >
                <code className="text-[10px] text-muted-foreground font-mono truncate">{doc._id}</code>
              </div>
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <div className="flex items-center justify-between pt-1" data-testid="pagination">
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-xs"
              disabled={page <= 1}
              onClick={() => onPageChange(page - 1)}
            >
              Prev
            </Button>
            <span className="text-xs text-muted-foreground">
              {page} / {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-xs"
              disabled={page >= totalPages}
              onClick={() => onPageChange(page + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
