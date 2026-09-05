'use client'

import { cn, Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Label } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Spinner, IconCheck } from '@sloughgpt/strui'
import { Tooltip, TooltipTrigger, TooltipContent } from '@sloughgpt/strui'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@sloughgpt/strui'
import { useDatasetImport } from '@/hooks/useDatasetImport'

interface DatasetImportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImportComplete: (datasetId: string) => void
}

const SOURCE_OPTIONS = [
  { value: 'github' as const, label: 'GitHub', description: 'Clone a repository', tip: 'Clones a GitHub repo and imports matching files. Requires git installed on the server.' },
  { value: 'huggingface' as const, label: 'HuggingFace', description: 'Download from HF Hub', tip: 'Downloads from HuggingFace Hub using the datasets library. Multi-config datasets auto-detect available configs.' },
  { value: 'isbn' as const, label: 'ISBN / Book', description: 'Search by title or ISBN', tip: 'Searches Open Library for books by title or ISBN, then tries Project Gutenberg for full text extraction.' },
  { value: 'kaggle' as const, label: 'Kaggle', description: 'Download from Kaggle', tip: 'Downloads from Kaggle. Requires Kaggle CLI installed and authenticated on the server.' },
  { value: 'csv' as const, label: 'CSV', description: 'Import CSV from URL', tip: 'Downloads a CSV file from a URL and converts it to JSONL format automatically.' },
  { value: 'url' as const, label: 'URL', description: 'Download from a URL', tip: 'Downloads content from any URL. The entire content becomes one JSONL record.' },
  { value: 'local' as const, label: 'Folder Path', description: 'Folder on this machine', tip: 'Imports files from a local folder. Supports .txt, .json, .csv, .pdf (via PyMuPDF), and more.' },
]

export function DatasetImportDialog({
  open,
  onOpenChange,
  onImportComplete,
}: DatasetImportDialogProps) {
  const di = useDatasetImport(onImportComplete, onOpenChange)

  return (
    <Dialog open={open} onOpenChange={di.handleOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Import Dataset</DialogTitle>
          <DialogDescription>
            Import training data from various sources
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4 overflow-y-auto flex-1 min-h-0">
          <fieldset>
            <legend className="sr-only">Import source</legend>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {SOURCE_OPTIONS.map((option) => (
                <Tooltip key={option.value}>
                  <TooltipTrigger asChild>
                    <Button
                      variant={di.source === option.value ? 'secondary' : 'outline'}
                      onClick={() => {
                        di.setSource(option.value)
                        di.clearError()
                        di.setSearchResults([])
                        di.setBookResults([])
                        di.setSelectedBook(null)
                        di.setSelectedRepo(null)
                      }}
                      className={cn('h-auto flex-col items-start p-3', di.source === option.value ? 'border-primary' : '')}
                      role="radio"
                      aria-checked={di.source === option.value}
                      aria-label={`${option.label}: ${option.description}`}
                    >
                      <div className="font-medium">{option.label}</div>
                      <div className="text-xs text-muted-foreground">{option.description}</div>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" variant="muted" className="max-w-64 text-xs">
                    {option.tip}
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </fieldset>

          {di.source === 'github' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="github-search">Search for a repository</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    id="github-search"
                    placeholder="e.g. shakespeare dataset, python tutorials"
                    value={di.url}
                    onChange={(e) => {
                      di.setUrl(e.target.value)
                      if (di.selectedRepo) di.setSelectedRepo(null)
                    }}
                    className="flex-1"
                  />
                  <Button type="button" variant="outline" size="sm" onClick={di.handleSearch} disabled={di.searching || !di.url.trim()} aria-busy={di.searching}>
                    {di.searching ? <Spinner className="w-4 h-4" /> : 'Search'}
                  </Button>
                </div>
              </div>
              {di.selectedRepo && (
                <div className="flex items-center gap-2 rounded-md bg-primary/5 border border-primary/20 px-3 py-2 text-sm">
                  <IconCheck className="w-4 h-4 text-primary shrink-0" />
                  <div className="min-w-0 flex-1">
                    <span className="font-medium truncate">{di.selectedRepo.full_name}</span>
                    {di.selectedRepo.description && (
                      <span className="text-muted-foreground ml-2 truncate">— {di.selectedRepo.description}</span>
                    )}
                  </div>
                  <span className="text-muted-foreground shrink-0">{di.selectedRepo.stars} ★</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-auto shrink-0 h-6 px-2 text-xs"
                    onClick={() => {
                      di.setSelectedRepo(null)
                      di.setUrl('')
                      di.setName('')
                    }}
                  >
                    Clear
                  </Button>
                </div>
              )}
            </div>
          )}

          {di.source === 'github' && di.searchResults.length > 0 && (
            <div className="max-h-48 overflow-y-auto rounded-md border" role="listbox" aria-label="Repository search results">
              {di.searchResults.map((repo) => (
                <Button
                  key={repo.id}
                  variant="ghost"
                  onClick={() => {
                    di.setSelectedRepo(repo)
                    di.setUrl(repo.full_name)
                    di.setName(repo.name)
                    di.setSearchResults([])
                  }}
                  className={cn('flex w-full items-start justify-between border-b px-3 py-2.5 text-left last:border-b-0', di.selectedRepo?.id === repo.id ? 'bg-primary/10' : '')}
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{repo.full_name}</div>
                    {repo.description && (
                      <div className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                        {repo.description}
                      </div>
                    )}
                    <div className="flex items-center gap-2 mt-1">
                      {repo.language && (
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0">{repo.language}</Badge>
                      )}
                      <span className="text-[10px] text-muted-foreground">{repo.stars} ★</span>
                    </div>
                  </div>
                </Button>
              ))}
            </div>
          )}

          {di.source === 'github' && !di.searching && di.searchResults.length === 0 && di.url.trim() && (
            <p className="text-xs text-muted-foreground">Type a search term and click Search to find repositories.</p>
          )}

          {di.source === 'isbn' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="isbn-search">Search by title or ISBN</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    id="isbn-search"
                    placeholder="e.g. The Hobbit or 9780547928227"
                    value={di.url}
                    onChange={(e) => {
                      di.setUrl(e.target.value)
                      if (di.selectedBook) di.setSelectedBook(null)
                    }}
                    className="flex-1"
                  />
                  <Button type="button" variant="outline" size="sm" onClick={di.handleSearch} disabled={di.searching || !di.url.trim()} aria-busy={di.searching}>
                    {di.searching ? <Spinner className="w-4 h-4" /> : 'Search'}
                  </Button>
                </div>
              </div>
              {di.selectedBook && (
                <div className="flex items-center gap-2 rounded-md bg-primary/5 border border-primary/20 px-3 py-2 text-sm">
                  <IconCheck className="w-4 h-4 text-primary shrink-0" />
                  <span className="font-medium truncate">{di.selectedBook.title}</span>
                  <span className="text-muted-foreground shrink-0">{di.selectedBook.isbn}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-auto shrink-0 h-6 px-2 text-xs"
                    onClick={() => {
                      di.setSelectedBook(null)
                      di.setUrl('')
                      di.setName('')
                    }}
                  >
                    Clear
                  </Button>
                </div>
              )}
            </div>
          )}

          {di.source === 'isbn' && di.bookResults.length > 0 && (
            <div className="max-h-48 overflow-y-auto rounded-md border" role="listbox" aria-label="Book search results">
              {di.bookResults.map((book) => (
                <Button
                  key={book.key}
                  variant="ghost"
                  onClick={() => {
                    di.setSelectedBook(book)
                    di.setName(book.title)
                    di.setUrl(book.isbn)
                    di.setBookResults([])
                  }}
                  className={cn('flex w-full items-center justify-between border-b px-3 py-2 text-left last:border-b-0', di.selectedBook?.key === book.key ? 'bg-primary/10' : '')}
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{book.title}</div>
                    <div className="text-xs text-muted-foreground">
                      {book.author} {book.year && `(${book.year})`}
                    </div>
                  </div>
                   <Badge variant="outline" className="shrink-0 ml-2">{book.isbn}</Badge>
                </Button>
              ))}
            </div>
          )}

          {di.source === 'isbn' && !di.searching && di.bookResults.length === 0 && di.url.trim() && !di.selectedBook && (
            <p className="text-xs text-muted-foreground">Type a title or ISBN and click Search to find books.</p>
          )}

          {di.source === 'huggingface' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="hf-id">HuggingFace Dataset ID</Label>
                <Input
                  id="hf-id"
                  placeholder="username/dataset-name"
                  value={di.datasetId}
                  onChange={(e) => di.setDatasetId(e.target.value)}
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Examples: <code className="font-mono text-[10px]">HuggingFaceH4/tinyshakespeare</code>, <code className="font-mono text-[10px]">HuggingFaceH4/ultrachat_200k</code>
                </p>
              </div>
            </div>
          )}

          {di.source === 'kaggle' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="kaggle-id">Kaggle Dataset ID</Label>
                <Input
                  id="kaggle-id"
                  placeholder="username/dataset-name"
                  value={di.kaggleDataset}
                  onChange={(e) => di.setKaggleDataset(e.target.value)}
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Examples: <code className="font-mono text-[10px]">heptapod/titanic</code>, <code className="font-mono text-[10px]">uciml/iris</code>, <code className="font-mono text-[10px]">rounakbanik/pokemon</code>
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Requires Kaggle CLI installed and authenticated (<code className="font-mono text-[10px]">kaggle config</code>)
                </p>
              </div>
            </div>
          )}

          {di.source === 'csv' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="csv-url">CSV File URL</Label>
                <Input
                  id="csv-url"
                  placeholder="https://example.com/data.csv"
                  value={di.url}
                  onChange={(e) => di.setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Import CSV file from URL - will be converted to JSONL format
              </p>
            </div>
          )}

          {di.source === 'url' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="url-input">URL</Label>
                <Input
                  id="url-input"
                  placeholder="https://example.com/data.txt"
                  value={di.url}
                  onChange={(e) => di.setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {di.source === 'local' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="local-path">Folder Path</Label>
                <p className="text-xs text-muted-foreground mb-2">
                  Absolute path to a folder. Matching files will be imported as a dataset.
                </p>
                <Input
                  id="local-path"
                  placeholder="/Users/mac/sloughGPT/datasets/my_data"
                  value={di.path}
                  onChange={(e) => {
                    di.setPath(e.target.value)
                    if (!di.name.trim() && e.target.value) {
                      const parts = e.target.value.replace(/\/$/, '').split('/')
                      di.setName(parts[parts.length - 1] || 'dataset')
                    }
                  }}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {(di.source === 'github' || di.source === 'local') && (
            <fieldset>
              <legend className="text-sm font-medium text-foreground mb-2">File Types (for code repos)</legend>
              <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Select file extensions">
                {['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.csv', '.pdf'].map((ext) => (
                  <Button
                    key={ext}
                    variant={di.extensions.includes(ext) ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => di.toggleExtension(ext)}
                    role="checkbox"
                    aria-checked={di.extensions.includes(ext)}
                    aria-label={`${ext} file type`}
                  >
                    {ext}
                  </Button>
                ))}
              </div>
            </fieldset>
          )}

          <div>
            <Label htmlFor="name">Dataset Name</Label>
            <Input
              id="name"
              placeholder="my-dataset"
              value={di.name}
              onChange={(e) => di.setName(e.target.value)}
              className="mt-1"
            />
          </div>

          {di.loading && (
            <div className="flex items-center gap-3 rounded-md bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
              <Spinner className="w-4 h-4" />
              <span>
                {di.source === 'local' ? 'Scanning folder...' :
                 di.source === 'github' ? 'Cloning repository...' :
                 di.source === 'huggingface' ? 'Downloading from HuggingFace...' :
                 'Importing...'}
              </span>
            </div>
          )}

          {di.error && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive" role="alert" aria-live="assertive">
              {di.error}
            </div>
          )}

          {di.success && (
            <div className="rounded-md bg-success/10 px-4 py-3 text-sm text-success" role="status" aria-live="polite">
              <div className="flex items-center gap-2 font-medium mb-1">
                <IconCheck className="w-4 h-4 shrink-0" />
                {di.success.message}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          {di.loading ? (
            <Button type="button" variant="outline" onClick={() => di.abortRef.current?.abort()}>
              Cancel Import
            </Button>
          ) : (
            <Button type="button" variant="outline" onClick={() => di.handleOpenChange(false)}>
              Cancel
            </Button>
          )}
          <Button type="button" onClick={di.handleImport} disabled={di.loading}>
            {di.loading ? 'Importing...' : 'Import'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
