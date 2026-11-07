import { describe, it, expect, vi, beforeEach } from 'vitest'

/**
 * notify() — the unified notification dispatch handle (card 20261107_064201).
 *
 * A THIN PURE DISPATCHER: no state, no timers, no policy of its own — it
 * forwards intents verbatim to the three existing tier stores, which stay
 * the state owners (banner-store single-slot, toast-store dedupe/TTL,
 * process dock projector). The surface lock below is deliberate: this
 * module is producers' ONLY door to notification tiers, and the dock's
 * `dispatchAction` (server-validated command execution) must never become
 * reachable through it — notifications are not commands.
 */

const { toastAdd, toastDismiss, toastClear, bannerShow, bannerDismiss, ingest } = vi.hoisted(
  () => ({
    toastAdd: vi.fn((..._args: unknown[]) => 'toast_1'),
    toastDismiss: vi.fn(),
    toastClear: vi.fn(),
    bannerShow: vi.fn((..._args: unknown[]) => 'banner_1'),
    bannerDismiss: vi.fn(),
    ingest: vi.fn(),
  }),
)

vi.mock('./toast-store', () => ({
  useToastStore: {
    getState: () => ({
      addToast: toastAdd,
      dismissToast: toastDismiss,
      clearToasts: toastClear,
    }),
  },
}))

vi.mock('./banner-store', () => ({
  useBannerStore: {
    getState: () => ({
      showBanner: bannerShow,
      dismissBanner: bannerDismiss,
    }),
  },
}))

vi.mock('./process-notifications', () => ({
  ingestProcessEvents: ingest,
}))

import { notify } from './notify'

beforeEach(() => {
  vi.clearAllMocks()
})

describe('notify — tier dispatch', () => {
  it('routes toast intents to the toast store, argument-for-argument', () => {
    const undo = () => {}
    notify.toast('Saved', 'success')
    expect(toastAdd).toHaveBeenLastCalledWith('Saved', 'success')
    notify.toast('Exported', 'info', 'full detail…', undo, 'export-key')
    expect(toastAdd).toHaveBeenLastCalledWith(
      'Exported',
      'info',
      'full detail…',
      undo,
      'export-key',
    )
  })

  it('mirrors the store return (toast id)', () => {
    expect(notify.toast('x')).toBe('toast_1')
  })

  it('routes banner intents to the banner store and passes the id through', () => {
    const id = notify.banner({
      tone: 'warning',
      title: 'Training interrupted',
      message: 'job 42 stopped',
    })
    expect(bannerShow).toHaveBeenCalledWith({
      tone: 'warning',
      title: 'Training interrupted',
      message: 'job 42 stopped',
    })
    expect(id).toBe('banner_1')
  })

  it('routes dismissal through the same handle', () => {
    notify.dismissToast('toast_9')
    expect(toastDismiss).toHaveBeenCalledWith('toast_9')
    notify.clearToasts()
    expect(toastClear).toHaveBeenCalledTimes(1)
    notify.dismissBanner('banner_9')
    expect(bannerDismiss).toHaveBeenCalledWith('banner_9')
  })

  it('routes process intents to the dock projector as a batch', () => {
    const event = { event_id: 'e1', kind: 'training', state: 'interrupted' } as never
    notify.process(event)
    expect(ingest).toHaveBeenCalledWith([event])
  })
})

describe('notify — surface lock (thin dispatcher, no commands)', () => {
  it('exposes exactly the six intent methods — no state, no extensions', () => {
    expect(Object.keys(notify).sort()).toEqual([
      'banner',
      'clearToasts',
      'dismissBanner',
      'dismissToast',
      'process',
      'toast',
    ])
  })

  it('can never reach the dock command surface (dispatchAction)', () => {
    expect(notify).not.toHaveProperty('dispatchAction')
    expect(notify).not.toHaveProperty('dispatch')
  })
})
