'use client'

import { useState, useRef } from 'react'
import { datasetController, type ImportSource, type GitHubRepo, type BookResult } from '@/lib/dataset-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { reportError } from '@/lib/error-reporter'
import { trackEvent } from '@/lib/dev-log'

interface UseDatasetImportReturn {
  source: ImportSource
  setSource: (s: ImportSource) => void
  url: string
  setUrl: (v: string) => void
  name: string
  setName: (v: string) => void
  datasetId: string
  setDatasetId: (v: string) => void
  kaggleDataset: string
  setKaggleDataset: (v: string) => void
  path: string
  setPath: (v: string) => void
  extensions: string[]
  toggleExtension: (ext: string) => void
  loading: boolean
  error: string | null
  success: { message: string } | null
  clearError: () => void
  searchResults: GitHubRepo[]
  bookResults: BookResult[]
  searching: boolean
  selectedBook: BookResult | null
  selectedRepo: GitHubRepo | null
  setSelectedRepo: (r: GitHubRepo | null) => void
  setSelectedBook: (b: BookResult | null) => void
  setSearchResults: (r: GitHubRepo[]) => void
  setBookResults: (r: BookResult[]) => void
  abortRef: React.RefObject<AbortController | null>
  handleSearch: () => Promise<void>
  handleImport: () => Promise<void>
  handleOpenChange: (open: boolean) => void
  resetForm: () => void
}

const DEFAULT_EXTENSIONS = ['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.csv', '.pdf']

export function useDatasetImport(
  onImportComplete: (datasetId: string) => void,
  onOpenChange: (open: boolean) => void
): UseDatasetImportReturn {
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

  const resetForm = () => {
    setUrl('')
    setName('')
    setDatasetId('')
    setKaggleDataset('')
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

  const toggleExtension = (ext: string) => {
    setExtensions(prev =>
      prev.includes(ext) ? prev.filter(e => e !== ext) : [...prev, ext]
    )
  }

  const selectRepo = (repo: GitHubRepo) => {
    setSelectedRepo(repo)
    setUrl(repo.full_name)
    setName(repo.name)
    setSearchResults([])
  }

  const handleSearch = async () => {
    if (!url.trim()) return
    setSearching(true)
    setError(null)
    try {
      if (source === 'isbn') {
        const result = await datasetController.searchBooks(url.trim())
        setBookResults(result.books)
      } else if (source === 'github') {
        const result = await datasetController.searchGitHubRepos(url.trim())
        setSearchResults(result.repos)
      }
    } catch (err) {
      const message = extractErrorMessage(err, 'Could not search')
      setError(message)
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
      const signal = ac.signal

      let result: { dataset_id: string; message: string }

      switch (source) {
        case 'github': {
          if (!url.trim()) {
            setError('GitHub URL is required')
            setLoading(false)
            return
          }
          const repoName = name.trim() || url.split('/').pop()?.replace('.git', '') || 'dataset'
          result = await datasetController.importFromGitHub({
            url: url.trim(),
            name: repoName,
            extensions,
          }, { signal })
          break
        }
        case 'huggingface': {
          if (!datasetId.trim()) {
            setError('HuggingFace dataset ID is required')
            setLoading(false)
            return
          }
          result = await datasetController.importFromHuggingFace({
            dataset_id: datasetId.trim(),
            name: name.trim() || datasetId.trim().split('/').pop() || 'dataset',
          }, { signal })
          break
        }
        case 'isbn': {
          const isbnValue = selectedBook?.isbn || url.trim()
          if (!isbnValue) {
            const msg = 'Enter a search term or ISBN'
            setError(msg)
            reportError(msg, 'dataset-import', { metadata: { source: 'isbn', action: 'import' } })
            setLoading(false)
            return
          }
          result = await datasetController.importFromISBN({
            isbn: isbnValue,
            name: name.trim() || selectedBook?.title || 'book-dataset',
          }, { signal })
          break
        }
        case 'kaggle': {
          if (!kaggleDataset.trim()) {
            setError('Kaggle dataset ID is required')
            setLoading(false)
            return
          }
          result = await datasetController.importFromKaggle({
            dataset: kaggleDataset.trim(),
            name: name.trim() || kaggleDataset.trim().split('/').pop() || 'dataset',
          }, { signal })
          break
        }
        case 'csv': {
          if (!url.trim()) {
            setError('CSV URL is required')
            setLoading(false)
            return
          }
          result = await datasetController.importFromCSV({
            url: url.trim(),
            name: name.trim() || 'csv-dataset',
          }, { signal })
          break
        }
        case 'url': {
          if (!url.trim()) {
            setError('URL is required')
            setLoading(false)
            return
          }
          result = await datasetController.importFromURL({
            url: url.trim(),
            name: name.trim() || 'downloaded-dataset',
          }, { signal })
          break
        }
        case 'local': {
          if (!path.trim()) {
            setError('Folder path is required')
            setLoading(false)
            return
          }
          result = await datasetController.importFromLocal({
            path: path.trim(),
            name: name.trim() || path.trim().split('/').pop() || 'local-dataset',
            extensions,
          }, { signal })
          break
        }
        default:
          throw new Error(`Unknown source: ${source}`)
      }

      setSuccess({ message: result.message || `Imported dataset` })
      trackEvent('dataset_imported', { source })
      onImportComplete(result.dataset_id)
    } catch (err) {
      if (ac.signal.aborted) return
      const message = extractErrorMessage(err, 'Could not import')
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return {
    source, setSource,
    url, setUrl,
    name, setName,
    datasetId, setDatasetId,
    kaggleDataset, setKaggleDataset,
    path, setPath,
    extensions, toggleExtension,
    loading, error, success,
    clearError: () => setError(null),
    searchResults, bookResults,
    searching,
    selectedBook, selectedRepo,
    setSelectedRepo, setSelectedBook,
    setSearchResults, setBookResults,
    abortRef,
    handleSearch, handleImport,
    handleOpenChange, resetForm,
  }
}
