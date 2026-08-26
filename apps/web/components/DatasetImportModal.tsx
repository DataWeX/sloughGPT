'use client'

import { useState, useRef, useEffect } from 'react'
import { datasetController, type ImportSource, type GitHubRepo, type BookResult, type ImportResponse } from '@/lib/dataset-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { reportError } from '@/lib/error-reporter'
import { cn, Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Label } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Spinner, IconCheck } from '@sloughgpt/strui'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@sloughgpt/strui'

interface DatasetImportModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImportComplete: (datasetId: string) => void
}

type SourceOption = {
  value: ImportSource
  label: string
  description: string
}

const SOURCE_OPTIONS: SourceOption[] = [
  { value: 'github', label: 'GitHub', description: 'Clone a repository' },
  { value: 'huggingface', label: 'HuggingFace', description: 'Download from HF Hub' },
  { value: 'isbn', label: 'ISBN / Book', description: 'Search by title or ISBN' },
  { value: 'kaggle', label: 'Kaggle', description: 'Download from Kaggle' },
  { value: 'csv', label: 'CSV', description: 'Import CSV from URL' },
  { value: 'url', label: 'URL', description: 'Download from a URL' },
  { value: 'local', label: 'Folder Path', description: 'Folder on this machine' },
]

const DEFAULT_EXTENSIONS = ['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.csv', '.pdf']

export function DatasetImportModal({
  open,
  onOpenChange,
  onImportComplete,
}: DatasetImportModalProps) {
  const [source, setSource] = useState<ImportSource>('github')
  const [url, setUrl] = useState('')
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [kaggleDataset, setKaggleDataset] = useState('')
  const [path, setPath] = useState('')
  const [extensions, setExtensions] = useState<string[]>(DEFAULT_EXTENSIONS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<{ message: string } | null>(null)
  const [searchResults, setSearchResults] = useState<GitHubRepo[]>([])
  const [bookResults, setBookResults] = useState<BookResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedBook, setSelectedBook] = useState<BookResult | null>(null)
  const [selectedRepo, setSelectedRepo] = useState<GitHubRepo | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!open) abortRef.current?.abort()
  }, [open])

  const resetForm = () => {
    setUrl('')
    setName('')
    setDatasetId('')
    setPath('')
    setExtensions(DEFAULT_EXTENSIONS)
    setError(null)
    setSuccess(null)
    setSearchResults([])
    setBookResults([])
    setSelectedBook(null)
    setSelectedRepo(null)
  }

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      abortRef.current?.abort()
      resetForm()
    }
    onOpenChange(newOpen)
  }

  const handleSearch = async () => {
    if (!url.trim()) return
    setSearching(true)
    setError(null)
    try {
      if (source === 'isbn') {
        const result = await datasetController.searchBooks(url.trim())
        setBookResults(result.books || [])
      } else {
        const result = await datasetController.searchGitHubRepos(url.trim())
        setSearchResults(result.repos || [])
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      const message = extractErrorMessage(err, 'Could not search')
      setError(message)
      reportError(message, 'dataset-import', { metadata: { source, action: 'search' } })
    } finally {
      setSearching(false)
    }
  }

  const handleImport = async () => {
    const ac = new AbortController()
    abortRef.current = ac
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      let result: ImportResponse
      const signal = ac.signal

      switch (source) {
        case 'github':
          if (!url.trim()) {
            throw new Error('GitHub URL is required')
          }
          const repoName = name.trim() || url.split('/').pop()?.replace('.git', '') || 'dataset'
          result = await datasetController.importFromGitHub({
            url: url.trim(),
            name: repoName,
            extensions,
          }, { signal })
          break

        case 'huggingface':
          if (!datasetId.trim()) {
            throw new Error('HuggingFace dataset ID is required')
          }
          result = await datasetController.importFromHuggingFace({
            dataset_id: datasetId.trim(),
            name: name.trim() || undefined,
          }, { signal })
          break

        case 'isbn':
          if (!url.trim()) {
            throw new Error('Enter a search term or ISBN')
          }
          if (!selectedBook && !url.trim().match(/^\d{10,13}$/)) {
            throw new Error('Select a book from the results, or enter a valid ISBN')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromISBN({
            isbn: selectedBook?.isbn || url.trim(),
            name: name.trim(),
          }, { signal })
          break

        case 'url':
          if (!url.trim()) {
            throw new Error('URL is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromURL({
            url: url.trim(),
            name: name.trim(),
          }, { signal })
          break

        case 'local':
          if (!path.trim()) {
            throw new Error('Folder path is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromLocal({
            path: path.trim(),
            name: name.trim(),
            extensions,
          }, { signal })
          break

        case 'kaggle':
          if (!kaggleDataset.trim()) {
            throw new Error('Kaggle dataset ID is required (e.g., zillow/zecon)')
          }
          result = await datasetController.importFromKaggle({
            dataset: kaggleDataset.trim(),
            name: name.trim() || undefined,
          }, { signal })
          break

        case 'csv':
          if (!url.trim()) {
            throw new Error('CSV URL is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromCSV({
            url: url.trim(),
            name: name.trim(),
          }, { signal })
          break
      }

      setSuccess({ message: result.message })
      setTimeout(() => {
        resetForm()
        onImportComplete(result.dataset_id)
        onOpenChange(false)
      }, 2000)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      const message = extractErrorMessage(err, 'Could not import')
      setError(message)
      reportError(message, 'dataset-import', { metadata: { source, action: 'import' } })
    } finally {
      setLoading(false)
    }
  }

  const toggleExtension = (ext: string) => {
    setExtensions((prev) =>
      prev.includes(ext) ? prev.filter((e) => e !== ext) : [...prev, ext]
    )
  }

  const selectRepo = (repo: GitHubRepo) => {
    setSelectedRepo(repo)
    setUrl(repo.url)
    setName(repo.name)
    setSearchResults([])
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
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
                <Button
                  key={option.value}
                  variant={source === option.value ? 'secondary' : 'outline'}
                  onClick={() => {
                    setSource(option.value)
                    setSearchResults([])
                    setBookResults([])
                    setSelectedBook(null)
                    setSelectedRepo(null)
                  }}
                  className={cn('h-auto flex-col items-start p-3', source === option.value ? 'border-primary' : '')}
                  role="radio"
                  aria-checked={source === option.value}
                  aria-label={`${option.label}: ${option.description}`}
                >
                  <div className="font-medium">{option.label}</div>
                  <div className="text-xs text-muted-foreground">{option.description}</div>
                </Button>
              ))}
            </div>
          </fieldset>

          {source === 'github' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="github-search">Search for a repository</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    id="github-search"
                    placeholder="e.g. shakespeare dataset, python tutorials"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value)
                      if (selectedRepo) setSelectedRepo(null)
                    }}
                    className="flex-1"
                  />
                  <Button type="button" variant="outline" size="sm" onClick={handleSearch} disabled={searching || !url.trim()} aria-busy={searching}>
                    {searching ? <Spinner className="w-4 h-4" /> : 'Search'}
                  </Button>
                </div>
              </div>
              {selectedRepo && (
                <div className="flex items-center gap-2 rounded-md bg-primary/5 border border-primary/20 px-3 py-2 text-sm">
                  <IconCheck className="w-4 h-4 text-primary shrink-0" />
                  <div className="min-w-0 flex-1">
                    <span className="font-medium truncate">{selectedRepo.full_name}</span>
                    {selectedRepo.description && (
                      <span className="text-muted-foreground ml-2 truncate">— {selectedRepo.description}</span>
                    )}
                  </div>
                  <span className="text-muted-foreground shrink-0">{selectedRepo.stars} ★</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-auto shrink-0 h-6 px-2 text-xs"
                    onClick={() => {
                      setSelectedRepo(null)
                      setUrl('')
                      setName('')
                    }}
                  >
                    Clear
                  </Button>
                </div>
              )}
            </div>
          )}

          {source === 'github' && searchResults.length > 0 && (
            <div className="max-h-48 overflow-y-auto rounded-md border" role="listbox" aria-label="Repository search results">
              {searchResults.map((repo) => (
                <Button
                  key={repo.id}
                  variant="ghost"
                  onClick={() => selectRepo(repo)}
                  className={cn('flex w-full items-start justify-between border-b px-3 py-2.5 text-left last:border-b-0', selectedRepo?.id === repo.id ? 'bg-primary/10' : '')}
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

          {source === 'github' && !searching && searchResults.length === 0 && url.trim() && (
            <p className="text-xs text-muted-foreground">Type a search term and click Search to find repositories.</p>
          )}

          {source === 'isbn' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="isbn-search">Search by title or ISBN</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    id="isbn-search"
                    placeholder="e.g. The Hobbit or 9780547928227"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value)
                      if (selectedBook) setSelectedBook(null)
                    }}
                    className="flex-1"
                  />
                  <Button type="button" variant="outline" size="sm" onClick={handleSearch} disabled={searching || !url.trim()} aria-busy={searching}>
                    {searching ? <Spinner className="w-4 h-4" /> : 'Search'}
                  </Button>
                </div>
              </div>
              {selectedBook && (
                <div className="flex items-center gap-2 rounded-md bg-primary/5 border border-primary/20 px-3 py-2 text-sm">
                  <IconCheck className="w-4 h-4 text-primary shrink-0" />
                  <span className="font-medium truncate">{selectedBook.title}</span>
                  <span className="text-muted-foreground shrink-0">{selectedBook.isbn}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-auto shrink-0 h-6 px-2 text-xs"
                    onClick={() => {
                      setSelectedBook(null)
                      setUrl('')
                      setName('')
                    }}
                  >
                    Clear
                  </Button>
                </div>
              )}
            </div>
          )}

          {source === 'isbn' && bookResults.length > 0 && (
            <div className="max-h-48 overflow-y-auto rounded-md border" role="listbox" aria-label="Book search results">
              {bookResults.map((book) => (
                <Button
                  key={book.key}
                  variant="ghost"
                  onClick={() => {
                    setSelectedBook(book)
                    setName(book.title)
                    setUrl(book.isbn)
                    setBookResults([])
                  }}
                  className={cn('flex w-full items-center justify-between border-b px-3 py-2 text-left last:border-b-0', selectedBook?.key === book.key ? 'bg-primary/10' : '')}
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate">{book.title}</div>
                    <div className="text-xs text-muted-foreground">
                      {book.author} {book.year && `(${book.year})`}
                    </div>
                  </div>
                  <Badge variant="secondary" className="shrink-0 ml-2">{book.isbn}</Badge>
                </Button>
              ))}
            </div>
          )}

          {source === 'isbn' && !searching && bookResults.length === 0 && url.trim() && !selectedBook && (
            <p className="text-xs text-muted-foreground">Type a title or ISBN and click Search to find books.</p>
          )}

          {source === 'huggingface' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="hf-id">HuggingFace Dataset ID</Label>
                <Input
                  id="hf-id"
                  placeholder="username/dataset-name"
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Examples: <code className="font-mono text-[10px]">HuggingFaceH4/tinyshakespeare</code>, <code className="font-mono text-[10px]">HuggingFaceH4/ultrachat_200k</code>
                </p>
              </div>
            </div>
          )}

          {source === 'kaggle' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="kaggle-id">Kaggle Dataset ID</Label>
                <Input
                  id="kaggle-id"
                  placeholder="username/dataset-name"
                  value={kaggleDataset}
                  onChange={(e) => setKaggleDataset(e.target.value)}
                  className="mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Examples: <code className="font-mono text-[10px]">zillow/zecon</code>, <code className="font-mono text-[10px]">nlp-datasets/tinyshakespeare</code>, <code className="font-mono text-[10px]">datasets/opensubtitles</code>
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Requires Kaggle CLI installed and authenticated (<code className="font-mono text-[10px]">kaggle config</code>)
                </p>
              </div>
            </div>
          )}

          {source === 'csv' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="csv-url">CSV File URL</Label>
                <Input
                  id="csv-url"
                  placeholder="https://example.com/data.csv"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Import CSV file from URL - will be converted to JSONL format
              </p>
            </div>
          )}

          {source === 'url' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="url-input">URL</Label>
                <Input
                  id="url-input"
                  placeholder="https://example.com/data.txt"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {source === 'local' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="local-path">Folder Path</Label>
                <p className="text-xs text-muted-foreground mb-2">
                  Absolute path to a folder. Matching files will be imported as a dataset.
                </p>
                <Input
                  id="local-path"
                  placeholder="/Users/mac/sloughGPT/datasets/my_data"
                  value={path}
                  onChange={(e) => {
                    setPath(e.target.value)
                    if (!name.trim() && e.target.value) {
                      const parts = e.target.value.replace(/\/$/, '').split('/')
                      setName(parts[parts.length - 1] || 'dataset')
                    }
                  }}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {(source === 'github' || source === 'local') && (
            <fieldset>
              <legend className="text-sm font-medium text-foreground mb-2">File Types (for code repos)</legend>
              <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Select file extensions">
                {['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.csv', '.pdf'].map((ext) => (
                  <Button
                    key={ext}
                    variant={extensions.includes(ext) ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => toggleExtension(ext)}
                    role="checkbox"
                    aria-checked={extensions.includes(ext)}
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
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1"
            />
          </div>

          {loading && (
            <div className="flex items-center gap-3 rounded-md bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
              <Spinner className="w-4 h-4" />
              <span>
                {source === 'local' ? 'Scanning folder...' :
                 source === 'github' ? 'Cloning repository...' :
                 source === 'huggingface' ? 'Downloading from HuggingFace...' :
                 'Importing...'}
              </span>
            </div>
          )}

          {error && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive" role="alert" aria-live="assertive">
              {error}
            </div>
          )}

          {success && (
            <div className="rounded-md bg-success/10 px-4 py-3 text-sm text-success" role="status" aria-live="polite">
              <div className="flex items-center gap-2 font-medium mb-1">
                <IconCheck className="w-4 h-4 shrink-0" />
                {success.message}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          {loading ? (
            <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}>
              Cancel Import
            </Button>
          ) : (
            <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
              Cancel
            </Button>
          )}
          <Button type="button" onClick={handleImport} disabled={loading}>
            {loading ? 'Importing...' : 'Import'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
