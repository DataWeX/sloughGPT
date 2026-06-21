import { describe, expect, it, beforeEach, vi, afterEach } from 'vitest'

// Mock console.debug to keep test output clean
beforeEach(() => { vi.spyOn(console, 'debug').mockImplementation(() => {}) })
afterEach(() => { vi.restoreAllMocks() })

// Dynamic imports so mocks are in place
import { useErrorStore, addGlobalError } from '../error-store'

describe('useErrorStore', () => {
  beforeEach(() => { useErrorStore.getState().clearErrors() })

  describe('addError', () => {
    it('adds an error with string message', () => {
      const id = useErrorStore.getState().addError('Something broke')
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err).toBeDefined()
      expect(err!.message).toBe('Something broke')
      expect(err!.title).toBe('Error')
      expect(err!.severity).toBe('error')
      expect(err!.dismissible).toBe(true)
    })

    it('adds an error from Error object', () => {
      const id = useErrorStore.getState().addError(new TypeError('bad type'))
      const errors = useErrorStore.getState().errors
      const err = errors.find(e => e.id === id)
      expect(err!.message).toBe('bad type')
      expect(err!.title).toBe('Type Error')
    })

    it('extracts message from object with detail field', () => {
      const id = useErrorStore.getState().addError({ detail: 'validation failed' })
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.message).toBe('validation failed')
    })

    it('extracts message from object with error field', () => {
      const id = useErrorStore.getState().addError({ error: 'internal error' })
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.message).toBe('internal error')
    })

    it('extracts message from object with msg field', () => {
      const id = useErrorStore.getState().addError({ msg: 'something happened' })
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.message).toBe('something happened')
    })

    it('falls back to stringified JSON for unknown objects', () => {
      const obj = { x: 1, y: 2 }
      const id = useErrorStore.getState().addError(obj)
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.message).toContain('x')
    })

    it('uses fallback for null input', () => {
      const id = useErrorStore.getState().addError(null)
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.message).toBe('null')
    })

    it('caps errors at 20 entries', () => {
      for (let i = 0; i < 25; i++) {
        useErrorStore.getState().addError(`error ${i}`)
      }
      expect(useErrorStore.getState().errors.length).toBe(20)
      expect(useErrorStore.getState().errors[0].message).toBe('error 24')
    })
  })

  describe('title extraction', () => {
    it('uses explicit title when provided', () => {
      const id = useErrorStore.getState().addError('msg', { title: 'Custom Title' })
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Custom Title')
    })

    it('detects 404 title from Error message', () => {
      const id = useErrorStore.getState().addError(new Error('404: not found'))
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Not Found')
    })

    it('detects 401 title from Error message', () => {
      const id = useErrorStore.getState().addError(new Error('401 Unauthorized'))
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Unauthorized')
    })

    it('detects timeout title from Error message', () => {
      const id = useErrorStore.getState().addError(new Error('timeout after 30s'))
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Timeout')
    })

    it('detects network error title from Error', () => {
      const id = useErrorStore.getState().addError(new Error('NetworkError: fetch failed'))
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Network Error')
    })

    it('detects ECONNREFUSED title from Error', () => {
      const id = useErrorStore.getState().addError(new Error('ECONNREFUSED 127.0.0.1:8000'))
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.title).toBe('Connection Refused')
    })
  })

  describe('severity', () => {
    it('uses explicit severity', () => {
      const id = useErrorStore.getState().addError('something', { severity: 'warning' })
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.severity).toBe('warning')
    })

    it('warns on 404', () => {
      const id = useErrorStore.getState().addError('not found')
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.severity).toBe('warning')
    })

    it('warns on timeout', () => {
      const id = useErrorStore.getState().addError('timeout after 30s')
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.severity).toBe('warning')
    })

    it('warns on network error', () => {
      const id = useErrorStore.getState().addError('connection refused')
      const err = useErrorStore.getState().errors.find(e => e.id === id)
      expect(err!.severity).toBe('warning')
    })
  })

  describe('dismissError', () => {
    it('removes an error by id', () => {
      const id = useErrorStore.getState().addError('test')
      expect(useErrorStore.getState().errors.length).toBe(1)
      useErrorStore.getState().dismissError(id)
      expect(useErrorStore.getState().errors.length).toBe(0)
    })
  })

  describe('clearErrors', () => {
    it('clears all errors', () => {
      useErrorStore.getState().addError('err1')
      useErrorStore.getState().addError('err2')
      useErrorStore.getState().clearErrors()
      expect(useErrorStore.getState().errors).toEqual([])
    })
  })

  describe('getErrors / hasErrors', () => {
    it('getErrors returns current list', () => {
      useErrorStore.getState().addError('test')
      expect(useErrorStore.getState().getErrors().length).toBe(1)
    })

    it('hasErrors returns false when empty', () => {
      expect(useErrorStore.getState().hasErrors()).toBe(false)
    })

    it('hasErrors returns true when errors exist', () => {
      useErrorStore.getState().addError('test')
      expect(useErrorStore.getState().hasErrors()).toBe(true)
    })
  })
})

describe('addGlobalError', () => {
  beforeEach(() => { useErrorStore.getState().clearErrors() })

  it('adds error and returns id', () => {
    const id = addGlobalError('global error', 'Chat')
    const err = useErrorStore.getState().errors.find(e => e.id === id)
    expect(err!.source).toBe('Chat')
    expect(err!.message).toBe('global error')
  })
})
