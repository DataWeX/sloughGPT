// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockListFineTuned: vi.fn(),
  mockDeleteFineTuned: vi.fn(),
  mockLoadFineTuned: vi.fn(),
  mockUnloadModel: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listFineTuned: mocks.mockListFineTuned,
    deleteFineTuned: mocks.mockDeleteFineTuned,
    loadFineTuned: mocks.mockLoadFineTuned,
  },
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    unloadModel: mocks.mockUnloadModel,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children }: any) => <div>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
  IconTrash: () => <span data-testid="icon-trash" />,
  IconRefresh: () => <span data-testid="icon-refresh" />,
  IconX: () => <span data-testid="icon-x" />,
  Checkbox: ({ checked, onCheckedChange, className, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} className={className} {...props} />
  ),
}))

import { FineTunedModelsCard } from './FineTunedModelsCard'

describe('FineTunedModelsCard', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows loading skeletons initially', () => {
    mocks.mockListFineTuned.mockReturnValue(new Promise(() => {}))
    render(<FineTunedModelsCard />)
    expect(screen.getByText('Fine-tuned models')).toBeDefined()
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no models', async () => {
    mocks.mockListFineTuned.mockResolvedValue([])
    render(<FineTunedModelsCard />)
    const el = await screen.findByText(/No fine-tuned models/)
    expect(el).toBeDefined()
  })

  it('renders model list', async () => {
    mocks.mockListFineTuned.mockResolvedValue([
      { name: 'my-model', model: 'gpt2', dataset: 'shakespeare', size_mb: 100, final_loss: 0.45, epochs: 10 },
    ])
    render(<FineTunedModelsCard />)
    expect(await screen.findByText('my-model')).toBeDefined()
    expect(screen.getByText(/gpt2/)).toBeDefined()
  })

  it('shows error state on fetch failure', async () => {
    mocks.mockListFineTuned.mockRejectedValue(new Error('network error'))
    render(<FineTunedModelsCard />)
    expect(await screen.findByText(/network error/)).toBeDefined()
    expect(screen.getByText('Retry')).toBeDefined()
  })

  it('renders refresh button', async () => {
    mocks.mockListFineTuned.mockResolvedValue([])
    render(<FineTunedModelsCard />)
    expect(screen.getByLabelText('Refresh fine-tuned models')).toBeDefined()
  })

  it('shows load button for inactive models', async () => {
    mocks.mockListFineTuned.mockResolvedValue([
      { name: 'm1', model: 'gpt2', dataset: '', size_mb: 0, final_loss: null, epochs: 0 },
    ])
    render(<FineTunedModelsCard />)
    expect(await screen.findByText('Load')).toBeDefined()
  })

  it('shows unload button for active model', async () => {
    mocks.mockListFineTuned.mockResolvedValue([
      { name: 'm1', model: 'gpt2', model_name: 'm1', dataset: '', size_mb: 0, final_loss: null, epochs: 0 },
    ])
    render(<FineTunedModelsCard activeModelId="m1" />)
    expect(await screen.findByText(/Unload/)).toBeDefined()
  })
})
