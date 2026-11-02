// @vitest-environment jsdom
/**
 * NotificationDock — the third-tier multi-slot renderer (card 20261102_014254).
 *
 * Renders the process-notification store: one card per event, each with the
 * process's structured action surface. Distinct from GlobalBanner (single
 * slot) — this dock stacks entries. Action clicks route through the real
 * dispatchAction with trainingController mocked underneath, so the
 * server-authority degradation (stale → "no longer available") is exercised
 * end-to-end through the store, not seeded by hand.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mockRecover = vi.hoisted(() => vi.fn())

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    recover: mockRecover,
    stop: vi.fn(),
  },
}))

import {
  clearProcessNotifications,
  ingestProcessEvents,
  type ProcessEvent,
} from '@/lib/process-notifications'
import { NotificationDock } from './NotificationDock'

const event = (overrides: Partial<ProcessEvent> = {}): ProcessEvent => ({
  event_id: 'training:j1:interrupted',
  kind: 'training',
  process_id: 'j1',
  event: 'interrupted',
  state: 'interrupted',
  actions: [
    { id: 'recover_now', label: 'Recover now', intent: 'recover' },
    { id: 'dismiss', label: 'Dismiss', intent: 'dismiss' },
  ],
  process_link: '/training/job/j1',
  title: 'Training interrupted',
  ...overrides,
})

beforeEach(() => {
  clearProcessNotifications()
  mockRecover.mockReset()
})

describe('NotificationDock', () => {
  it('renders nothing when there are no notifications', () => {
    const { container } = render(<NotificationDock />)
    expect(container.firstChild).toBeNull()
  })

  it('renders an event title and the process link', () => {
    ingestProcessEvents([event()])
    render(<NotificationDock />)
    expect(screen.getByText('Training interrupted')).toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: /view job/i }),
    ).toHaveAttribute('href', '/training/job/j1')
  })

  it('renders every action from the contract as a control', () => {
    ingestProcessEvents([event()])
    render(<NotificationDock />)
    expect(
      screen.getByRole('button', { name: 'Recover now' }),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Dismiss' })).toBeInTheDocument()
  })

  it('multi-slot: renders two events as two cards', () => {
    ingestProcessEvents([
      event(),
      event({
        event_id: 'training:j2:interrupted',
        process_id: 'j2',
        process_link: '/training/job/j2',
      }),
    ])
    render(<NotificationDock />)
    expect(screen.getAllByText('Training interrupted')).toHaveLength(2)
  })

  it('Recover click dispatches through the controller with the process id', async () => {
    mockRecover.mockResolvedValue({ status: 'recovered' })
    ingestProcessEvents([event()])
    render(<NotificationDock />)

    fireEvent.click(screen.getByRole('button', { name: 'Recover now' }))

    await waitFor(() => expect(mockRecover).toHaveBeenCalledWith('j1'))
  })

  it('shows the dispatched confirmation after a successful action', async () => {
    mockRecover.mockResolvedValue({ status: 'recovered' })
    ingestProcessEvents([event()])
    render(<NotificationDock />)

    fireEvent.click(screen.getByRole('button', { name: 'Recover now' }))

    await waitFor(() =>
      expect(screen.getByText(/dispatched/i)).toBeInTheDocument(),
    )
  })

  it('degrades to "no longer available" when the server refuses (stale action)', async () => {
    // Server re-validates against CURRENT state and refuses a stale click.
    mockRecover.mockRejectedValue(new Error("Job status is 'running'"))
    ingestProcessEvents([event()])
    render(<NotificationDock />)

    fireEvent.click(screen.getByRole('button', { name: 'Recover now' }))

    await waitFor(() =>
      expect(screen.getByText(/no longer available/i)).toBeInTheDocument(),
    )
    // The refused action is no longer offered as clickable.
    expect(
      screen.queryByRole('button', { name: 'Recover now' }),
    ).not.toBeInTheDocument()
  })

  it('renders the skip reason when the contract carries one', () => {
    ingestProcessEvents([
      event({
        event: 'auto_resume_skipped',
        title: 'Auto-resume skipped',
        reason: 'no loadable checkpoint',
      }),
    ])
    render(<NotificationDock />)
    expect(screen.getByText(/no loadable checkpoint/i)).toBeInTheDocument()
  })

  it('Dismiss click removes the entry via dispatchAction', async () => {
    ingestProcessEvents([event()])
    render(<NotificationDock />)

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))

    await waitFor(() =>
      expect(screen.queryByText('Training interrupted')).not.toBeInTheDocument(),
    )
    expect(mockRecover).not.toHaveBeenCalled()
  })
})
