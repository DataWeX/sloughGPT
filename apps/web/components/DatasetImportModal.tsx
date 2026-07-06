'use client'

import { useState } from 'react'
import { datasetController, type ImportSource, type GitHubRepo, type BookResult, type ImportResponse } from '@/lib/dataset-controller'
import { Button } from '@sloughgpt/strui'
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
  { value: 'local', label: 'Server Path', description: 'Folder on this machine' },
]

const DEFAULT_EXTENSIONS = ['.py', '.js', '.ts', '.md', '.txt', '.json', '.pdf']

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
  }

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
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
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  const handleImport = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      let result: ImportResponse

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
          })
          break

        case 'huggingface':
          if (!datasetId.trim()) {
            throw new Error('HuggingFace dataset ID is required')
          }
          result = await datasetController.importFromHuggingFace({
            dataset_id: datasetId.trim(),
            name: name.trim() || undefined,
          })
          break

        case 'isbn':
          if (bookResults.length > 0) {
            throw new Error('Select a book from the search results')
          }
          if (!url.trim()) {
            throw new Error('Enter a search term or ISBN')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromISBN({
            isbn: url.trim(),
            name: name.trim(),
          })
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
          })
          break

        case 'local':
          if (!path.trim()) {
            throw new Error('Server path is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await datasetController.importFromLocal({
            path: path.trim(),
            name: name.trim(),
            extensions,
          })
          break

        case 'kaggle':
          if (!kaggleDataset.trim()) {
            throw new Error('Kaggle dataset ID is required (e.g., zillow/zecon)')
          }
          result = await datasetController.importFromKaggle({
            dataset: kaggleDataset.trim(),
            name: name.trim() || undefined,
          })
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
          })
          break
      }

      setSuccess({ message: result.message })
      setTimeout(() => {
        resetForm()
        onImportComplete(result.dataset_id)
        onOpenChange(false)
      }, 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed')
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
    setUrl(repo.url)
    setName(repo.name)
    setSearchResults([])
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Import Dataset</DialogTitle>
          <DialogDescription>
            Import training data from various sources
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
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
                  }}
                  className={`h-auto flex-col items-start p-3 ${
                    source === option.value ? 'border-primary' : ''
                  }`}
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
                <Label htmlFor="github-url">GitHub Repository URL</Label>
                <Input
                  id="github-url"
                  placeholder="https://github.com/user/repo"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>

              {searchResults.length > 0 && (
                <div className="max-h-40 overflow-y-auto rounded-md border" role="listbox" aria-label="Search results">
                  {searchResults.map((repo) => (
                    <Button
                      key={repo.id}
                      variant="ghost"
                      onClick={() => selectRepo(repo)}
                      className="flex w-full items-center justify-between border-b px-3 py-2 text-left last:border-b-0"
                    >
                      <div>
                        <div className="font-medium">{repo.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {repo.full_name}
                        </div>
                      </div>
                      <Badge variant="secondary">{repo.stars} stars</Badge>
                    </Button>
                  ))}
                </div>
              )}

              <Button type="button" variant="outline" size="sm" onClick={handleSearch} disabled={searching} aria-busy={searching}>
                {searching ? 'Searching...' : 'Search'}
              </Button>
            </div>
          )}

          {source === 'isbn' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="isbn-search">Search by title or ISBN</Label>
                <Input
                  id="isbn-search"
                  placeholder="e.g. The Hobbit or 9780547928227"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
              <Button type="button" variant="outline" size="sm" onClick={handleSearch} disabled={searching} aria-busy={searching}>
                {searching ? 'Searching...' : 'Search'}
              </Button>
            </div>
          )}

          {source === 'isbn' && bookResults.length > 0 && (
            <div className="max-h-40 overflow-y-auto rounded-md border">
              {bookResults.map((book) => (
                <Button
                  key={book.key}
                  variant="ghost"
                  onClick={() => {
                    setName(book.title)
                    setUrl(book.isbn)
                    setBookResults([])
                  }}
                  className="flex w-full items-center justify-between border-b px-3 py-2 text-left last:border-b-0"
                >
                  <div>
                    <div className="font-medium">{book.title}</div>
                    <div className="text-xs text-muted-foreground">
                      {book.author} {book.year && `(${book.year})`}
                    </div>
                  </div>
                  <Badge variant="secondary">{book.isbn}</Badge>
                </Button>
              ))}
            </div>
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
              </div>
              <p className="text-xs text-muted-foreground">
                Requires Kaggle CLI installed and authenticated
              </p>
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
                <Label htmlFor="local-path">Server Path</Label>
                <p className="text-xs text-muted-foreground mb-2">
                  Absolute path to a folder on this server. Matching files will be imported as a dataset.
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
                {source === 'local' ? 'Scanning server path...' :
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
          <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" onClick={handleImport} disabled={loading}>
            {loading ? 'Importing...' : 'Import'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
