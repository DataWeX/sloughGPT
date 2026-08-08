import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockImportFromLocal: vi.fn(),
  mockImportFromGitHub: vi.fn(),
  mockImportFromHuggingFace: vi.fn(),
  mockImportFromURL: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    importFromLocal: mocks.mockImportFromLocal,
    importFromGitHub: mocks.mockImportFromGitHub,
    importFromHuggingFace: mocks.mockImportFromHuggingFace,
    importFromURL: mocks.mockImportFromURL,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => {
  const tabApi: { value: string; onChange: (v: string) => void } = {
    value: 'local',
    onChange: () => {},
  }
  return {
    Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
      open ? <div>{children}</div> : null,
    DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Spinner: () => <div data-testid="spinner" />,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: (props: any) => <input {...props} />,
    Tabs: ({ value, onValueChange, children }: any) => {
      tabApi.value = value
      tabApi.onChange = onValueChange
      return <div>{children}</div>
    },
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ value, children }: any) => (
      <button data-tab={value} onClick={() => tabApi.onChange(value)}>{children}</button>
    ),
    TabsContent: ({ value, children }: any) =>
      tabApi.value === value ? <div>{children}</div> : null,
  }
})

import DatasetInlineImportModal from './DatasetInlineImportModal'

const SUCCESS = { dataset_id: 'ds-1', message: 'Imported 5 files' }

describe('DatasetInlineImportModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the dialog when open', () => {
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={() => {}} />)
    expect(screen.getByText('Import Dataset')).toBeDefined()
    expect(screen.getByPlaceholderText('Dataset name (optional)')).toBeDefined()
    expect(screen.getByText('Local Path')).toBeDefined()
    expect(screen.getByText('GitHub')).toBeDefined()
    expect(screen.getByText('HuggingFace')).toBeDefined()
    expect(screen.getByText('URL')).toBeDefined()
  })

  it('renders nothing when closed', () => {
    const { container } = render(<DatasetInlineImportModal open={false} onOpenChange={() => {}} onImported={() => {}} />)
    expect(container.innerHTML).toBe('')
  })

  it('imports from a local path with the default name', async () => {
    mocks.mockImportFromLocal.mockResolvedValueOnce(SUCCESS)
    const onImported = vi.fn()
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={onImported} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: '/data' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockImportFromLocal).toHaveBeenCalledWith({
      path: '/data',
      name: 'imported_dataset',
    }))
    expect(mocks.mockAddToast).toHaveBeenCalledWith('Imported 5 files', 'success')
    expect(onImported).toHaveBeenCalled()
    expect(screen.getByText('Imported successfully')).toBeDefined()
    expect(screen.getByText('Imported 5 files')).toBeDefined()
  })

  it('passes a custom dataset name when provided', async () => {
    mocks.mockImportFromLocal.mockResolvedValueOnce(SUCCESS)
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText('Dataset name (optional)'), { target: { value: 'my-ds' } })
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: '/data' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockImportFromLocal).toHaveBeenCalledWith({
      path: '/data',
      name: 'my-ds',
    }))
  })

  it('imports from a GitHub URL on the GitHub tab', async () => {
    mocks.mockImportFromGitHub.mockResolvedValueOnce(SUCCESS)
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={() => {}} />)
    fireEvent.click(screen.getByText('GitHub'))
    fireEvent.change(screen.getByPlaceholderText('https://github.com/user/repo'), { target: { value: 'https://github.com/a/b' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockImportFromGitHub).toHaveBeenCalledWith({
      url: 'https://github.com/a/b',
      name: 'imported_dataset',
    }))
  })

  it('leaves the name undefined for HuggingFace imports', async () => {
    mocks.mockImportFromHuggingFace.mockResolvedValueOnce(SUCCESS)
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={() => {}} />)
    fireEvent.click(screen.getByText('HuggingFace'))
    fireEvent.change(screen.getByPlaceholderText('username/dataset-name'), { target: { value: 'org/ds' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockImportFromHuggingFace).toHaveBeenCalledWith({
      dataset_id: 'org/ds',
      name: undefined,
    }))
  })

  it('shows an error toast when the active field is empty', async () => {
    const onImported = vi.fn()
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={onImported} />)
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Fill in the required field', 'error'))
    expect(mocks.mockImportFromLocal).not.toHaveBeenCalled()
    expect(onImported).not.toHaveBeenCalled()
  })

  it('shows an error toast and skips onImported when the import fails', async () => {
    mocks.mockImportFromLocal.mockRejectedValueOnce(new Error('boom'))
    const onImported = vi.fn()
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={onImported} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: '/data' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Import failed', 'error'))
    expect(onImported).not.toHaveBeenCalled()
  })

  it('disables the button and shows a spinner while importing', async () => {
    let resolveImport!: (v: typeof SUCCESS) => void
    mocks.mockImportFromLocal.mockReturnValue(new Promise(r => { resolveImport = r }))
    render(<DatasetInlineImportModal open onOpenChange={() => {}} onImported={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: '/data' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    const importingButton = await waitFor(() => screen.getByRole('button', { name: /Importing/ }))
    expect((importingButton as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByTestId('spinner')).toBeDefined()

    resolveImport(SUCCESS)
    await waitFor(() => expect(screen.getByText('Imported successfully')).toBeDefined())
  })

  it('closes the dialog via the Done button and resets state', async () => {
    mocks.mockImportFromLocal.mockResolvedValueOnce(SUCCESS)
    const onOpenChange = vi.fn()
    render(<DatasetInlineImportModal open onOpenChange={onOpenChange} onImported={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText('/path/to/dataset/folder'), { target: { value: '/data' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))
    await waitFor(() => expect(screen.getByText('Imported successfully')).toBeDefined())

    fireEvent.click(screen.getByRole('button', { name: 'Done' }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
