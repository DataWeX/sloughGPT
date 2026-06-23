// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const { mockTrainingJobs, mockAddToast } = vi.hoisted(() => ({
  mockTrainingJobs: {
    listWebhooks: vi.fn(),
    webhookStats: vi.fn().mockResolvedValue({}),
    createWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    testWebhook: vi.fn(),
  },
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: mockTrainingJobs,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/dev-log', () => ({
  devDebug: vi.fn(),
}))

import { WebhookManager } from './WebhookManager'

describe('WebhookManager', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockTrainingJobs.listWebhooks.mockResolvedValue([])
  })

  afterEach(cleanup)

  it('shows empty state when no webhooks', async () => {
    render(<WebhookManager />)
    const empty = await screen.findByText('No webhooks configured')
    expect(empty).toBeDefined()
  })

  it('shows add webhook form when + clicked', async () => {
    render(<WebhookManager />)
    fireEvent.click(screen.getByText('+ Add Webhook'))
    expect(screen.getByPlaceholderText('https://example.com/webhook')).toBeDefined()
  })

  it('calls createWebhook when add submitted', async () => {
    render(<WebhookManager />)
    fireEvent.click(screen.getByText('+ Add Webhook'))
    fireEvent.change(screen.getByPlaceholderText('https://example.com/webhook'), { target: { value: 'https://hook.example.com' } })
    fireEvent.click(screen.getByText('Add Webhook'))
    expect(mockTrainingJobs.createWebhook).toHaveBeenCalledWith('https://hook.example.com', ['training.completed'])
  })

  it('renders webhook list with items', async () => {
    mockTrainingJobs.listWebhooks.mockResolvedValue([
      { id: 'wh1', url: 'https://hook.example.com', events: ['training.completed'], is_active: true, description: '', created_at: '' },
    ])
    render(<WebhookManager />)
    const url = await screen.findByText('https://hook.example.com')
    expect(url).toBeDefined()
  })

  it('shows event badges for webhooks', async () => {
    mockTrainingJobs.listWebhooks.mockResolvedValue([
      { id: 'wh1', url: 'https://hook.example.com', events: ['training.completed', 'job.failed'], is_active: true, description: '', created_at: '' },
    ])
    render(<WebhookManager />)
    const badge = await screen.findByText('completed')
    expect(badge).toBeDefined()
  })

  it('calls deleteWebhook on delete', async () => {
    window.confirm = vi.fn(() => true)
    mockTrainingJobs.listWebhooks.mockResolvedValue([
      { id: 'wh1', url: 'https://hook.example.com', events: ['training.completed'], is_active: true, description: '', created_at: '' },
    ])
    render(<WebhookManager />)
    const deleteBtn = await screen.findByText('✕')
    fireEvent.click(deleteBtn)
    expect(mockTrainingJobs.deleteWebhook).toHaveBeenCalledWith('wh1')
  })

  it('calls testWebhook on test', async () => {
    mockTrainingJobs.listWebhooks.mockResolvedValue([
      { id: 'wh1', url: 'https://hook.example.com', events: ['training.completed'], is_active: true, description: '', created_at: '' },
    ])
    render(<WebhookManager />)
    const testBtn = await screen.findByText('Test')
    fireEvent.click(testBtn)
    expect(mockTrainingJobs.testWebhook).toHaveBeenCalledWith('https://hook.example.com')
  })
})
