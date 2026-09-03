import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { useErrorStore, addGlobalError } from './error-store'

describe('error-store', () => {
  beforeEach(() => { useErrorStore.getState().clearErrors() })
  afterEach(() => { useErrorStore.getState().clearErrors() })

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
    const words = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet', 'Kilo', 'Lima', 'Mike', 'November', 'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform', 'Victor', 'Whiskey', 'Xray', 'Yankee']
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
})
