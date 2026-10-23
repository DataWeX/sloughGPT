import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { useErrorStore, addGlobalError } from './error-store'
import { useBannerStore } from './banner-store'
import { useToastStore } from './toast-store'

describe('error-store', () => {
  beforeEach(() => {
    useErrorStore.getState().clearErrors()
    useErrorStore.getState().clearStateEvents()
    useBannerStore.getState().clearBanners()
    useToastStore.getState().clearToasts()
  })
  afterEach(() => {
    useErrorStore.getState().clearErrors()
    useErrorStore.getState().clearStateEvents()
  })

  it('starts with no errors', () => {
    expect(useErrorStore.getState().hasErrors()).toBe(false)
    expect(useErrorStore.getState().errors).toEqual([])
  })

  it('adds string error', () => {
    useErrorStore.getState().addError('Something broke')
    expect(useErrorStore.getState().errors[0].message).toBe('Something broke')
  })

  it('adds Error object', () => {
    useErrorStore.getState().addError(new Error('fail'))
    expect(useErrorStore.getState().errors[0].message).toBe('fail')
  })

  it('adds error with detail field', () => {
    useErrorStore.getState().addError({ detail: 'Not found' })
    expect(useErrorStore.getState().errors[0].message).toBe('Not found')
  })

  it('generates title from HTTP error object', () => {
    useErrorStore.getState().addError({ message: '404 Not Found' })
    expect(useErrorStore.getState().errors[0].title).toBe('Not Found')
  })

  it('generates title from Error name', () => {
    useErrorStore.getState().addError({ message: 'HTTP 401 Unauthorized' })
    expect(useErrorStore.getState().errors[0].title).toBe('Unauthorized')
  })

  it('generates title for network error objects', () => {
    useErrorStore.getState().addError({ message: 'Network error occurred' })
    expect(useErrorStore.getState().errors[0].title).toBe('Network Error')
  })

  it('returns generic Error title for unknown', () => {
    useErrorStore.getState().addError('Something weird happened')
    expect(useErrorStore.getState().errors[0].title).toBe('Error')
  })

  it('uses explicit title override', () => {
    useErrorStore.getState().addError('Something', { title: 'Custom Title' })
    expect(useErrorStore.getState().errors[0].title).toBe('Custom Title')
  })

  it('sets severity based on message content', () => {
    useErrorStore.getState().addError('not found')
    expect(useErrorStore.getState().errors[0].severity).toBe('warning')
  })

  it('sets explicit severity', () => {
    useErrorStore.getState().addError('Something', { severity: 'info' })
    expect(useErrorStore.getState().errors[0].severity).toBe('info')
  })

  it('sets source and requestId', () => {
    useErrorStore.getState().addError('fail', { source: 'Chat', requestId: 'r1' })
    expect(useErrorStore.getState().errors[0].source).toBe('Chat')
    expect(useErrorStore.getState().errors[0].requestId).toBe('r1')
  })

  it('dismissError removes error', () => {
    const id = useErrorStore.getState().addError('fail')
    expect(useErrorStore.getState().hasErrors()).toBe(true)
    useErrorStore.getState().dismissError(id)
    expect(useErrorStore.getState().hasErrors()).toBe(false)
  })

  it('clearErrors removes all errors', () => {
    useErrorStore.getState().addError('A')
    useErrorStore.getState().addError('B')
    useErrorStore.getState().clearErrors()
    expect(useErrorStore.getState().errors).toEqual([])
  })

  it('caps errors at 20', () => {
    const words = [
      'Alpha',
      'Bravo',
      'Charlie',
      'Delta',
      'Echo',
      'Foxtrot',
      'Golf',
      'Hotel',
      'India',
      'Juliet',
      'Kilo',
      'Lima',
      'Mike',
      'November',
      'Oscar',
      'Papa',
      'Quebec',
      'Romeo',
      'Sierra',
      'Tango',
      'Uniform',
      'Victor',
      'Whiskey',
      'Xray',
      'Yankee',
    ]
    for (let i = 0; i < 25; i++) useErrorStore.getState().addError(`Fail ${words[i]}`)
    expect(useErrorStore.getState().errors.length).toBe(20)
    expect(useErrorStore.getState().errors[0].message).toBe('Fail Yankee')
    expect(useErrorStore.getState().errors[19].message).toBe('Fail Foxtrot')
  })

  it('addGlobalError adds error via store getState', () => {
    addGlobalError('Global fail', 'SourceA')
    expect(useErrorStore.getState().errors[0].source).toBe('SourceA')
  })

  it('extracts Error.name as title', () => {
    const err = new TypeError('unexpected type')
    useErrorStore.getState().addError(err)
    expect(useErrorStore.getState().errors[0].title).toBe('Type Error')
  })

  it('handles object without message field', () => {
    useErrorStore.getState().addError({ someField: 'value' })
    expect(useErrorStore.getState().errors[0].message).toContain('someField')
  })

  it('starts with no state events', () => {
    expect(useErrorStore.getState().getStateEvents()).toEqual([])
  })

  it('logs connection transition with from/to', () => {
    useErrorStore.getState().logStateEvent('connection_status_changed', {
      kind: 'connection',
      from: 'connecting',
      to: 'connected',
    })
    const [e] = useErrorStore.getState().getStateEvents()
    expect(e.event).toBe('connection_status_changed')
    expect(e.kind).toBe('connection')
    expect(e.from).toBe('connecting')
    expect(e.to).toBe('connected')
  })

  it('infers api kind from api_ prefix', () => {
    useErrorStore.getState().logStateEvent('api_connection_changed', { from: 'a', to: 'b' })
    expect(useErrorStore.getState().getStateEvents()[0].kind).toBe('api')
  })

  it('dedups consecutive identical transitions', () => {
    useErrorStore.getState().logStateEvent('sse_open', { kind: 'sse' })
    useErrorStore.getState().logStateEvent('sse_open', { kind: 'sse' })
    expect(useErrorStore.getState().getStateEvents().length).toBe(1)
  })

  it('caps state events at 200', () => {
    for (let i = 0; i < 220; i++) {
      useErrorStore.getState().logStateEvent(`health_ping_${i}`, { kind: 'health' })
    }
    expect(useErrorStore.getState().getStateEvents().length).toBe(200)
    expect(useErrorStore.getState().getStateEvents()[0].event).toBe('health_ping_219')
  })

  it('clearStateEvents empties buffer but keeps errors', () => {
    useErrorStore.getState().addError('keep me')
    useErrorStore.getState().logStateEvent('startup_stage_changed', { kind: 'startup' })
    useErrorStore.getState().clearStateEvents()
    expect(useErrorStore.getState().getStateEvents()).toEqual([])
    expect(useErrorStore.getState().errors.length).toBe(1)
  })

  it.each([
    ['training_started', 'training'],
    ['checkpoint_loaded', 'training'],
    ['dataset_imported', 'training'],
    ['knowledge_added', 'training'],
    ['auth_login', 'auth'],
    ['workspace_switch', 'auth'],
    ['session_created', 'chat'],
    ['stream_started', 'chat'],
    ['chat_mode_changed', 'chat'],
    ['conversion_stage_changed', 'chat'],
    ['soul_changed', 'chat'],
    ['model_loaded', 'model'],
    ['vm_booted', 'system'],
    ['shell_command_started', 'system'],
    ['webhook_created', 'system'],
    ['operation_cancelled', 'system'],
    ['error_lifecycle_initialized', 'system'],
    ['route_changed', 'ui'],
    ['locale_changed', 'ui'],
    ['theme_changed', 'ui'],
    ['settings_changed', 'ui'],
    ['feedback_recorded', 'ui'],
    ['startup_stage_changed', 'startup'],
    ['overlay_shown', 'overlay'],
    ['sse_open', 'sse'],
    ['health_fallback_error', 'health'],
    ['api_connection_changed', 'api'],
    ['connection_status_changed', 'connection'],
    ['something_totally_new', 'system'],
  ] as const)('infers kind %s → %s', (event, kind) => {
    useErrorStore.getState().logStateEvent(event)
    expect(useErrorStore.getState().getStateEvents()[0].kind).toBe(kind)
  })

  it('explicit kind wins over inference', () => {
    useErrorStore.getState().logStateEvent('training_started', { kind: 'ui' })
    expect(useErrorStore.getState().getStateEvents()[0].kind).toBe('ui')
  })

  describe('funnel fan-out', () => {
    it('logs an error_reported state event with taxonomy', () => {
      useErrorStore.getState().addError(new Error('boom'), {
        source: 'Test',
        surface: 'page',
        journeyId: 'ux/kanban',
      })
      const evt = useErrorStore.getState().getStateEvents()[0]
      expect(evt.event).toBe('error_reported')
      expect(evt.kind).toBe('system')
      expect(evt.data).toMatchObject({ surface: 'page', journeyId: 'ux/kanban' })
    })

    it('raises the global banner for page-surface errors', () => {
      useErrorStore.getState().addError(new Error('board load failed'), {
        source: 'Board',
        surface: 'page',
        title: 'Board failed to load',
      })
      const banner = useBannerStore.getState().banners[0]
      expect(banner).toBeTruthy()
      expect(banner.tone).toBe('destructive')
      expect(banner.title).toBe('Board failed to load')
      expect(banner.key).toMatch(/^err_/)
    })

    it('raises the banner for global-surface errors (unhandled runtime)', () => {
      useErrorStore.getState().addError(new Error('unhandled'), {
        surface: 'global',
        title: 'Runtime Error',
      })
      expect(useBannerStore.getState().banners[0]?.title).toBe('Runtime Error')
    })

    it('does NOT banner for component-surface errors', () => {
      useErrorStore.getState().addError(new Error('widget'), { surface: 'component' })
      expect(useBannerStore.getState().banners).toHaveLength(0)
    })

    it('toasts only when the producer asks', () => {
      useErrorStore.getState().addError(new Error('copied? no'), {
        surface: 'component',
        toast: true,
        title: 'Copy failed',
      })
      expect(useToastStore.getState().toasts[0]?.message).toBe('Copy failed')
      expect(useBannerStore.getState().banners).toHaveLength(0)
    })

    it('banner opt-out overrides the page-surface default', () => {
      useErrorStore.getState().addError(new Error('quiet'), {
        surface: 'page',
        banner: false,
        title: 'Quiet page error',
      })
      expect(useBannerStore.getState().banners).toHaveLength(0)
    })

    it('deduped repeats do not re-fan-out (one banner per fingerprint window)', () => {
      const opts = { source: 'Board', surface: 'page' as const }
      useErrorStore.getState().addError(new Error('same failure'), opts)
      useErrorStore.getState().addError(new Error('same failure'), opts)
      expect(useBannerStore.getState().banners).toHaveLength(1)
      const reportEvents = useErrorStore
        .getState()
        .getStateEvents()
        .filter((e) => e.event === 'error_reported')
      expect(reportEvents).toHaveLength(1)
    })
  })
})
