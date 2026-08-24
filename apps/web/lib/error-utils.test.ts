import { describe, it, expect } from 'vitest'
import { extractErrorMessage, formatToastError } from './error-utils'

describe('extractErrorMessage', () => {
  it('extracts message from Error object', () => {
    expect(extractErrorMessage(new Error('test failure'))).toBe('test failure')
  })

  it('extracts message from TypeError', () => {
    expect(extractErrorMessage(new TypeError('type mismatch'))).toBe('type mismatch')
  })

  it('returns string value directly', () => {
    expect(extractErrorMessage('raw string error')).toBe('raw string error')
  })

  it('returns default fallback for undefined', () => {
    expect(extractErrorMessage(undefined)).toBe('Unknown error')
  })

  it('returns default fallback for null', () => {
    expect(extractErrorMessage(null)).toBe('Unknown error')
  })

  it('converts number to string', () => {
    expect(extractErrorMessage(42)).toBe('42')
  })

  it('serializes object to JSON when no known error properties', () => {
    expect(extractErrorMessage({ code: 500 })).toBe('{"code":500}')
  })

  it('returns custom fallback when provided', () => {
    expect(extractErrorMessage(undefined, 'Custom fallback')).toBe('Custom fallback')
  })

  it('returns custom fallback for null', () => {
    expect(extractErrorMessage(null, 'Oops')).toBe('Oops')
  })

  it('returns message from Error even when custom fallback is provided', () => {
    expect(extractErrorMessage(new Error('real error'), 'fallback')).toBe('real error')
  })

  it('handles empty string error', () => {
    expect(extractErrorMessage('')).toBe('')
  })

  it('handles Error with empty message', () => {
    expect(extractErrorMessage(new Error(''))).toBe('')
  })
})

describe('formatToastError', () => {
  it('formats Error with prefix', () => {
    expect(formatToastError(new Error('fail'), 'Operation failed')).toBe('Operation failed: fail')
  })

  it('formats string error with prefix', () => {
    expect(formatToastError('bad input', 'Validation')).toBe('Validation: bad input')
  })

  it('formats unknown value with prefix', () => {
    expect(formatToastError(null, 'Error')).toBe('Error: Unknown error')
  })

  it('formats undefined with prefix', () => {
    expect(formatToastError(undefined, 'Something broke')).toBe('Something broke: Unknown error')
  })

  it('handles empty prefix', () => {
    expect(formatToastError(new Error('err'), '')).toBe(': err')
  })
})
