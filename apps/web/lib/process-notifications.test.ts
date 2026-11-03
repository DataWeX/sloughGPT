/**
 * Process-notification projector store (card 20261102_014254).
 *
 * Third notification tier: multi-slot, process-backed, action-first.
 * The store projects ProcessEvents (see training/process_events.py) into
 * dock notifications — deduped by event_id (the contract's stable key),
 * loaded from the REST snapshot at boot (SSE cannot deliver events that
 * fired before a client connected), and dispatching actions through the
 * controller layer only (never raw fetch), with server-authority
 * degradation: a refused action marks the notification "no longer
 * available" instead of pretending it succeeded.
 *
 * Distinct from banner-store (single-slot journey blocker) and
 * toast-store (transient confirmation): this dock is multi-slot and the
 * entries persist until their process state changes or the user acts.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const mockRecover = vi.hoisted(() => vi.fn())
const mockStop = vi.hoisted(() => vi.fn())
const mockGet = vi.hoisted(() => vi.fn())

vi.mock('@/lib/http-client', () => ({
  apiGet: mockGet,
  apiPost: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    recover: mockRecover,
    stop: mockStop,
  },
}))

import {
  type ProcessEvent,
  clearProcessNotifications,
  dispatchAction,
  getProcessNotifications,
  ingestProcessEvents,
  loadProcessSnapshot,
  useProcessNotifications,
} from './process-notifications'

const interruptedEvent = (overrides: Partial<ProcessEvent> = {}): ProcessEvent => ({
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
  contract_version: 1,
  ...overrides,
})

beforeEach(() => {
  clearProcessNotifications()
  mockRecover.mockReset()
  mockStop.mockReset()
  mockGet.mockReset()
})

describe('ingestProcessEvents — dedupe by event_id', () => {
  it('adds a new event', () => {
    ingestProcessEvents([interruptedEvent()])
    expect(getProcessNotifications()).toHaveLength(1)
  })

  it('does not duplicate the same event_id (snapshot vs live transition)', () => {
    ingestProcessEvents([interruptedEvent()])
    ingestProcessEvents([interruptedEvent()])
    expect(getProcessNotifications()).toHaveLength(1)
  })

  it('keeps distinct event_ids as separate notifications (multi-slot)', () => {
    ingestProcessEvents([
      interruptedEvent(),
      interruptedEvent({
        event_id: 'training:j2:interrupted',
        process_id: 'j2',
        process_link: '/training/job/j2',
      }),
    ])
    expect(getProcessNotifications()).toHaveLength(2)
  })

  it('updates an existing event in place when its state moves on', () => {
    ingestProcessEvents([interruptedEvent()])
    ingestProcessEvents([
      interruptedEvent({
        event: 'auto_train_started',
        state: 'running',
        actions: [
          { id: 'cancel', label: 'Cancel', intent: 'cancel' },
          { id: 'view_logs', label: 'View logs', intent: 'navigate' },
        ],
        title: 'Training auto-resumed',
      }),
    ])
    const notes = getProcessNotifications()
    expect(notes).toHaveLength(1)
    expect(notes[0].event).toBe('auto_train_started')
    expect(notes[0].actions.map((a) => a.id)).toEqual(['cancel', 'view_logs'])
  })

  it('drops settled events (empty action surface)', () => {
    ingestProcessEvents([
      interruptedEvent({
        event: 'interrupted',
        state: 'interrupted',
        actions: [],
      }),
    ])
    expect(getProcessNotifications()).toHaveLength(0)
  })
})

describe('loadProcessSnapshot — boot source', () => {
  it('fetches /recovery/events and ingests the events', async () => {
    mockGet.mockResolvedValue({
      contract_version: 1,
      count: 1,
      events: [interruptedEvent()],
    })

    await loadProcessSnapshot()

    expect(mockGet).toHaveBeenCalledWith('/recovery/events')
    expect(getProcessNotifications()).toHaveLength(1)
  })

  it('tolerates a failed snapshot without throwing', async () => {
    mockGet.mockRejectedValue(new Error('offline'))

    await expect(loadProcessSnapshot()).resolves.toBeUndefined()
    expect(getProcessNotifications()).toHaveLength(0)
  })

  it('ignores an unknown contract_version (fail-closed, no mis-render)', async () => {
    mockGet.mockResolvedValue({ contract_version: 99, events: [interruptedEvent()] })

    await loadProcessSnapshot()

    expect(getProcessNotifications()).toHaveLength(0)
  })
})

describe('dispatchAction — server is the authority', () => {
  it('recover action routes through trainingController.recover', async () => {
    mockRecover.mockResolvedValue({ status: 'recovered' })
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:j1:interrupted', 'recover_now')

    expect(mockRecover).toHaveBeenCalledWith('j1')
  })

  it('cancel action routes through trainingController.stop', async () => {
    mockStop.mockResolvedValue(undefined)
    ingestProcessEvents([
      interruptedEvent({
        event_id: 'training:j1:auto_train_started',
        event: 'auto_train_started',
        state: 'running',
        actions: [{ id: 'cancel', label: 'Cancel', intent: 'cancel' }],
        title: 'Training auto-resumed',
      }),
    ])

    await dispatchAction('training:j1:auto_train_started', 'cancel')

    expect(mockStop).toHaveBeenCalledWith('j1')
  })

  it('marks the notification dispatched after success', async () => {
    mockRecover.mockResolvedValue({ status: 'recovered' })
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:j1:interrupted', 'recover_now')

    expect(getProcessNotifications()[0].dispatched).toBe(true)
  })

  it('degrades to "no longer available" when the server refuses (stale action)', async () => {
    // Server re-validates against CURRENT state: job already running -> 400.
    mockRecover.mockRejectedValue(new Error("Job status is 'running'"))
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:j1:interrupted', 'recover_now')

    const note = getProcessNotifications()[0]
    expect(note.dispatched).toBe(false)
    expect(note.stale).toBe(true)
  })

  it('unknown action id is a no-op (never dispatches garbage)', async () => {
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:j1:interrupted', 'not_a_real_action')

    expect(mockRecover).not.toHaveBeenCalled()
  })

  it('dismiss intent removes the notification without any network call', async () => {
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:j1:interrupted', 'dismiss')

    expect(getProcessNotifications()).toHaveLength(0)
    expect(mockRecover).not.toHaveBeenCalled()
  })

  it('navigate intent does not dispatch a mutation', async () => {
    ingestProcessEvents([
      interruptedEvent({
        actions: [{ id: 'view_logs', label: 'View logs', intent: 'navigate' }],
      }),
    ])

    await dispatchAction('training:j1:interrupted', 'view_logs')

    expect(mockRecover).not.toHaveBeenCalled()
    expect(getProcessNotifications()).toHaveLength(1)
  })

  it('unknown event_id is a no-op', async () => {
    ingestProcessEvents([interruptedEvent()])

    await dispatchAction('training:ghost:interrupted', 'recover_now')

    expect(mockRecover).not.toHaveBeenCalled()
  })
})

describe('useProcessNotifications — zustand surface', () => {
  it('exposes getState with the notifications array', () => {
    ingestProcessEvents([interruptedEvent()])
    expect(useProcessNotifications.getState().notifications).toHaveLength(1)
  })

  it('notifies subscribers on change', () => {
    const seen: number[] = []
    const unsub = useProcessNotifications.subscribe((s) =>
      seen.push(s.notifications.length),
    )
    ingestProcessEvents([interruptedEvent()])
    unsub()
    expect(seen).toContain(1)
  })
})
