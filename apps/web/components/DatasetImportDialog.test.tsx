import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Tooltip: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Spinner: () => <div data-testid="spinner" />,
  IconCheck: () => <div data-testid="icon-check" />,
  Button: ({ children, onClick, variant, size, ...rest }: any) => (
    <button onClick={onClick} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label {...props}>{children}</label>,
  Badge: ({ children, variant }: any) => <span data-testid="badge" data-variant={variant}>{children}</span>,
}))

const mocks = vi.hoisted(() => ({
  mockImportFromGitHub: vi.fn(),
  mockImportFromHuggingFace: vi.fn(),
  mockImportFromISBN: vi.fn(),
  mockImportFromURL: vi.fn(),
  mockImportFromLocal: vi.fn(),
  mockImportFromKaggle: vi.fn(),
  mockImportFromCSV: vi.fn(),
  mockSearchGitHubRepos: vi.fn(),
  mockSearchBooks: vi.fn(),
  mockReportError: vi.fn(),
}))

vi.mock('@/lib/error-reporter', () => ({
  reportError: mocks.mockReportError,
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    importFromGitHub: mocks.mockImportFromGitHub,
    importFromHuggingFace: mocks.mockImportFromHuggingFace,
    importFromISBN: mocks.mockImportFromISBN,
    importFromURL: mocks.mockImportFromURL,
    importFromLocal: mocks.mockImportFromLocal,
    importFromKaggle: mocks.mockImportFromKaggle,
    importFromCSV: mocks.mockImportFromCSV,
    searchGitHubRepos: mocks.mockSearchGitHubRepos,
    searchBooks: mocks.mockSearchBooks,
  },
}))

import { DatasetImportDialog } from './DatasetImportDialog'

describe('DatasetImportDialog', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders dialog when open', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByText('Import Dataset')).toBeDefined()
  })

  it('does not render when closed', () => {
    const { container } = render(
      <DatasetImportDialog open={false} onOpenChange={() => {}} onImportComplete={() => {}} />,
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders source option buttons', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByText('GitHub')).toBeDefined()
    expect(screen.getByText('HuggingFace')).toBeDefined()
    expect(screen.getByText('Kaggle')).toBeDefined()
    expect(screen.getByText('URL')).toBeDefined()
    expect(screen.getByText('Folder Path')).toBeDefined()
  })

  it('renders name input', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByLabelText('Dataset Name')).toBeDefined()
  })

  it('shows extension toggles for GitHub source', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByText('.py')).toBeDefined()
    expect(screen.getByText('.js')).toBeDefined()
    expect(screen.getByText('.ts')).toBeDefined()
  })

  it('shows GitHub URL input when GitHub source selected', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByLabelText('Search for a repository')).toBeDefined()
  })

  it('shows HuggingFace ID input when HF source selected', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('HuggingFace'))
    expect(screen.getByLabelText('HuggingFace Dataset ID')).toBeDefined()
  })

  it('shows Server Path input when Local source selected', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Folder Path'))
    expect(screen.getByLabelText('Folder Path')).toBeDefined()
  })

  it('shows Kaggle dataset ID input when Kaggle source selected', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Kaggle'))
    expect(screen.getByLabelText('Kaggle Dataset ID')).toBeDefined()
  })

  it('shows CSV URL input when CSV source selected', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('CSV'))
    expect(screen.getByLabelText('CSV File URL')).toBeDefined()
  })

  it('shows error for empty GitHub URL on Import', async () => {
    const { rerender } = render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText('GitHub URL is required')).toBeDefined()
    })
  })

  it('shows error for empty HuggingFace dataset ID', async () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('HuggingFace'))
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText('HuggingFace dataset ID is required')).toBeDefined()
    })
  })

  it('shows error for empty server path', async () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Folder Path'))
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText('Folder path is required')).toBeDefined()
    })
  })

  it('shows error for empty Kaggle dataset ID', async () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Kaggle'))
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText(/Kaggle dataset ID is required/)).toBeDefined()
    })
  })

  it('shows error and reports it for empty ISBN search', async () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('ISBN / Book'))
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText('Enter a search term or ISBN')).toBeDefined()
    })
    expect(mocks.mockReportError).toHaveBeenCalledWith(
      'Enter a search term or ISBN',
      'dataset-import',
      expect.objectContaining({ metadata: expect.objectContaining({ source: 'isbn', action: 'import' }) }),
    )
  })

  it('shows loading state during GitHub import', async () => {
    mocks.mockImportFromGitHub.mockImplementation(() => new Promise(() => {}))
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    const urlInput = screen.getByLabelText('Search for a repository')
    fireEvent.change(urlInput, { target: { value: 'https://github.com/user/repo' } })
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => {
      expect(screen.getByText('Importing...')).toBeDefined()
    })
  })

  it('shows success message after import', async () => {
    mocks.mockImportFromGitHub.mockResolvedValue({ dataset_id: 'my-dataset', message: 'Imported 5 files (12,345 chars)' })
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    const urlInput = screen.getByLabelText('Search for a repository')
    fireEvent.change(urlInput, { target: { value: 'https://github.com/user/repo' } })
    fireEvent.click(screen.getByText('Import'))
    expect(await screen.findByText('Imported 5 files (12,345 chars)')).toBeDefined()
  })

  it('calls onImportComplete after successful import', async () => {
    const onComplete = vi.fn()
    mocks.mockImportFromGitHub.mockResolvedValue({ dataset_id: 'my-dataset', message: 'done' })
    vi.useFakeTimers()
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={onComplete} />)
    const urlInput = screen.getByLabelText('Search for a repository')
    fireEvent.change(urlInput, { target: { value: 'https://github.com/user/repo' } })
    fireEvent.click(screen.getByText('Import'))
    await vi.waitFor(() => {
      expect(mocks.mockImportFromGitHub).toHaveBeenCalled()
    })
    vi.advanceTimersByTime(2000)
    expect(onComplete).toHaveBeenCalledWith('my-dataset')
    vi.useRealTimers()
  })

  it('calls onOpenChange(false) on Cancel button', () => {
    const onClose = vi.fn()
    render(<DatasetImportDialog open={true} onOpenChange={onClose} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Cancel'))
    expect(onClose).toHaveBeenCalledWith(false)
  })

  it('searches for GitHub repos', async () => {
    mocks.mockSearchGitHubRepos.mockResolvedValue({
      repos: [
        { id: 1, name: 'test-repo', full_name: 'user/test-repo', url: 'https://github.com/user/test-repo', stars: 42, language: 'Python', description: 'A test repo' },
      ],
    })
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    const urlInput = screen.getByLabelText('Search for a repository')
    fireEvent.change(urlInput, { target: { value: 'user/test-repo' } })
    fireEvent.click(screen.getByText('Search'))
    await waitFor(() => {
      expect(screen.getByText('user/test-repo')).toBeDefined()
    })
  })

  it('selects a GitHub repo from search results', async () => {
    mocks.mockSearchGitHubRepos.mockResolvedValue({
      repos: [
        { id: 1, name: 'test-repo', full_name: 'user/test-repo', url: 'https://github.com/user/test-repo', stars: 42, language: 'Python', description: 'A test repo' },
      ],
    })
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    const urlInput = screen.getByLabelText('Search for a repository')
    fireEvent.change(urlInput, { target: { value: 'user/test-repo' } })
    fireEvent.click(screen.getByText('Search'))
    await waitFor(() => {
      expect(screen.getByText('user/test-repo')).toBeDefined()
    })
    fireEvent.click(screen.getByText('user/test-repo'))
    await waitFor(() => {
      expect(screen.queryByRole('listbox', { name: 'Repository search results' })).not.toBeInTheDocument()
    })
  })

  it('auto-populates name from path for local source', () => {
    render(<DatasetImportDialog open={true} onOpenChange={() => {}} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('Folder Path'))
    const pathInput = screen.getByLabelText('Folder Path')
    fireEvent.change(pathInput, { target: { value: '/Users/test/my_data' } })
    expect(screen.getByDisplayValue('my_data')).toBeDefined()
  })

  it('resets form on close', () => {
    const onClose = vi.fn()
    const { rerender } = render(
      <DatasetImportDialog open={true} onOpenChange={onClose} onImportComplete={() => {}} />,
    )
    fireEvent.click(screen.getByText('Folder Path'))
    fireEvent.change(screen.getByLabelText('Folder Path'), { target: { value: '/some/path' } })
    rerender(<DatasetImportDialog open={true} onOpenChange={onClose} onImportComplete={() => {}} />)
    fireEvent.click(screen.getByText('GitHub'))
    expect(screen.getByLabelText('Search for a repository')).toBeDefined()
  })
})
