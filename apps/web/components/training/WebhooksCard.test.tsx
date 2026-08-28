import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Button: ({ children, onClick, disabled, className }: any) => (
      <button onClick={onClick} disabled={disabled} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Input: ({ value, onChange, placeholder, className }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} />
    ),
    Label: ({ children }: any) => <label>{children}</label>,
    Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick, className }: any) => <button onClick={onClick} className={className}>{children}</button>,
    AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <p>{children}</p>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    Checkbox: ({ checked, onCheckedChange, className, ...props }: any) => (
      <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} className={className} {...props} />
    ),
  }
})

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, confirmLabel, title }: any) => open ? (
    <div data-testid="alert-dialog">
      <span>{title}</span>
      <button onClick={onConfirm}>{confirmLabel}</button>
    </div>
  ) : null,
}))

const mocks = vi.hoisted(() => ({
  listWebhooks: vi.fn(),
  createWebhook: vi.fn(),
  deleteWebhook: vi.fn(),
  testWebhook: vi.fn(),
  getWebhookDeliveries: vi.fn(),
  getWebhookRetryQueue: vi.fn(),
  getWebhookDeadLetters: vi.fn(),
  webhookStats: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listWebhooks: mocks.listWebhooks,
    createWebhook: mocks.createWebhook,
    deleteWebhook: mocks.deleteWebhook,
    testWebhook: mocks.testWebhook,
    getWebhookDeliveries: mocks.getWebhookDeliveries,
    getWebhookRetryQueue: mocks.getWebhookRetryQueue,
    getWebhookDeadLetters: mocks.getWebhookDeadLetters,
    webhookStats: mocks.webhookStats,
  },
}))

import { WebhooksCard } from './WebhooksCard'

const mockToast = vi.fn()

function renderCard() {
  return render(<WebhooksCard addToast={mockToast} />)
}

beforeEach(() => {
  vi.clearAllMocks()
  cleanup()
  mocks.listWebhooks.mockResolvedValue([])
  mocks.getWebhookDeliveries.mockResolvedValue([])
  mocks.getWebhookRetryQueue.mockResolvedValue({ retries: [] })
  mocks.getWebhookDeadLetters.mockResolvedValue({ dead_letters: [] })
  mocks.webhookStats.mockResolvedValue({ total_deliveries: 0, successful_deliveries: 0, failed_deliveries: 0, success_rate: '0%' })
})

function getAllButtons(name: string): HTMLElement[] {
  return screen.getAllByRole('button').filter(b => b.textContent?.trim() === name)
}

describe('WebhooksCard', () => {
  it('renders title', async () => {
    renderCard()
    expect(screen.getByText(/Webhooks/)).toBeTruthy()
  })

  it('shows empty state when no webhooks', async () => {
    renderCard()
    await waitFor(() => {
      expect(screen.getAllByText(/No webhooks yet/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows loading skeletons', async () => {
    mocks.listWebhooks.mockImplementation(() => new Promise(() => {}))
    renderCard()
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('displays webhooks', async () => {
    mocks.listWebhooks.mockResolvedValue([
      { id: 'w1', url: 'https://example.com/hook', events: ['training.completed'], active: true },
    ])
    renderCard()
    await waitFor(() => {
      expect(screen.getAllByText('https://example.com/hook').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('creates a webhook', async () => {
    mocks.createWebhook.mockResolvedValue({ id: 'w1' })
    renderCard()
    await waitFor(() => { expect(screen.getAllByText(/No webhooks yet/).length).toBeGreaterThanOrEqual(1) })

    const input = screen.getAllByPlaceholderText('https://example.com/webhook')[0]
    fireEvent.change(input, { target: { value: 'https://my.app/hook' } })
    fireEvent.click(getAllButtons('Add webhook')[0])

    await waitFor(() => {
      expect(mocks.createWebhook).toHaveBeenCalledWith('https://my.app/hook', ['training.completed'])
    })
    expect(mockToast).toHaveBeenCalledWith('Webhook added', 'success')
  })

  it('shows error toast on create failure', async () => {
    mocks.createWebhook.mockRejectedValue(new Error('fail'))
    renderCard()
    await waitFor(() => { expect(screen.getAllByText(/No webhooks yet/).length).toBeGreaterThanOrEqual(1) })

    const input = screen.getAllByPlaceholderText('https://example.com/webhook')[0]
    fireEvent.change(input, { target: { value: 'https://my.app/hook' } })
    fireEvent.click(getAllButtons('Add webhook')[0])

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not add webhook', 'error')
    })
  })

  it('tests a webhook', async () => {
    mocks.testWebhook.mockResolvedValue({})
    mocks.listWebhooks.mockResolvedValue([
      { id: 'w1', url: 'https://example.com/hook', events: ['training.completed'], active: true },
    ])
    renderCard()
    await waitFor(() => { expect(screen.getAllByText('https://example.com/hook').length).toBeGreaterThanOrEqual(1) })

    fireEvent.click(getAllButtons('Test')[0])
    await waitFor(() => {
      expect(mocks.testWebhook).toHaveBeenCalledWith('https://example.com/hook')
    })
    expect(mockToast).toHaveBeenCalledWith('Webhook test sent', 'success')
  })

  it('shows error toast on test failure', async () => {
    mocks.testWebhook.mockRejectedValue(new Error('fail'))
    mocks.listWebhooks.mockResolvedValue([
      { id: 'w1', url: 'https://example.com/hook', events: ['training.completed'], active: true },
    ])
    renderCard()
    await waitFor(() => { expect(screen.getAllByText('https://example.com/hook').length).toBeGreaterThanOrEqual(1) })

    fireEvent.click(getAllButtons('Test')[0])
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not webhook test', 'error')
    })
  })

  it('opens delete dialog and deletes', async () => {
    mocks.deleteWebhook.mockResolvedValue({})
    mocks.listWebhooks.mockResolvedValue([
      { id: 'w1', url: 'https://example.com/hook', events: ['training.completed'], active: true },
    ])
    renderCard()
    await waitFor(() => { expect(screen.getAllByText('https://example.com/hook').length).toBeGreaterThanOrEqual(1) })

    fireEvent.click(getAllButtons('Delete')[0])
    expect(screen.getByTestId('alert-dialog')).toBeTruthy()

    fireEvent.click(screen.getByText('Delete Webhook'))
    await waitFor(() => {
      expect(mocks.deleteWebhook).toHaveBeenCalledWith('w1')
    })
    expect(mockToast).toHaveBeenCalledWith('Webhook deleted', 'success')
  })

  it('displays delivery history', async () => {
    mocks.listWebhooks.mockResolvedValue([
      { id: 'w1', url: 'https://example.com/hook', events: ['training.completed'], active: true },
    ])
    mocks.getWebhookDeliveries.mockResolvedValue([
      { id: 'd1', event: 'training.completed', status: 200, success: true, delivered_at: '2026-08-20T12:00:00Z' },
    ])
    renderCard()
    await waitFor(() => { expect(screen.getAllByText('https://example.com/hook').length).toBeGreaterThanOrEqual(1) })

    const row = screen.getAllByText('https://example.com/hook')[0].closest('[role="button"]')!
    fireEvent.click(row)
    await waitFor(() => {
      expect(screen.getAllByText(/Delivered/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays retry queue', async () => {
    mocks.listWebhooks.mockResolvedValue([])
    mocks.getWebhookRetryQueue.mockResolvedValue({ retries: [
      { delivery_id: 'd1', webhook_id: 'w1', event: 'training.completed', attempt_count: 2, next_retry_at: 1724150400 },
    ] })
    renderCard()
    await waitFor(() => {
      expect(screen.getByText(/Retry queue/)).toBeTruthy()
    })
  })
})
