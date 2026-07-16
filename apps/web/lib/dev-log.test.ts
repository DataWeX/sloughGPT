import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => { (process.env as Record<string, string>).NODE_ENV = 'development' })

import { WebLogger, devDebug, logger } from './dev-log'

beforeEach(() => {
  vi.spyOn(console, 'debug').mockImplementation(() => {})
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('WebLogger', () => {
  it('emits debug to console.debug', () => {
    const log = new WebLogger('test', 'debug')
    log.debug('hello')
    expect(console.debug).toHaveBeenCalledWith('[test]', 'hello', expect.objectContaining({ level: 'debug', logger: 'test', message: 'hello' }))
  })

  it('does not emit debug when level is warning', () => {
    const log = new WebLogger('test', 'warning')
    log.debug('should not appear')
    expect(console.debug).not.toHaveBeenCalled()
  })

  it('emits warning correctly', () => {
    const log = new WebLogger('test', 'debug')
    log.warning('warn')
    expect(console.warn).toHaveBeenCalledWith('[test]', 'warn', expect.any(Object))
  })

  it('emits error correctly', () => {
    const log = new WebLogger('test', 'debug')
    log.error('boom')
    expect(console.error).toHaveBeenCalled()
  })

  it('info calls console.log', () => {
    const log = new WebLogger('test', 'info')
    log.info('hello')
    expect(console.log).toHaveBeenCalledWith('[test]', 'hello', expect.any(Object))
  })

  it('critical calls console.error', () => {
    const log = new WebLogger('test', 'debug')
    log.critical('fatal')
    expect(console.error).toHaveBeenCalledWith('[test]', 'fatal', expect.objectContaining({ level: 'critical' }))
  })

  it('error accepts exception in opts', () => {
    const log = new WebLogger('test', 'debug')
    log.error('err', { exception: 'TypeError' })
    const record = (console.error as ReturnType<typeof vi.fn>).mock.calls[0][2]
    expect(record.exception).toBe('TypeError')
  })

  it('error passes context fields', () => {
    const log = new WebLogger('test', 'debug')
    log.error('err', { session_id: 'abc' })
    const record = (console.error as ReturnType<typeof vi.fn>).mock.calls[0][2]
    expect(record.context.session_id).toBe('abc')
  })

  it('setContext merges context into every record', () => {
    const log = new WebLogger('test', 'debug')
    log.setContext({ request_id: 'r1' })
    log.info('msg')
    const record = (console.log as ReturnType<typeof vi.fn>).mock.calls[0][2]
    expect(record.context.request_id).toBe('r1')
  })

  it('clearContext removes all context', () => {
    const log = new WebLogger('test', 'debug')
    log.setContext({ request_id: 'r1' })
    log.clearContext()
    log.info('msg')
    const record = (console.log as ReturnType<typeof vi.fn>).mock.calls[0][2]
    expect(record.context.request_id).toBeUndefined()
  })

  it('child creates a sub-logger with prefixed name', () => {
    const parent = new WebLogger('man.web', 'debug')
    const child = parent.child('chat')
    expect(child.name).toBe('man.web.chat')
    child.info('hi')
    expect(console.log).toHaveBeenCalledWith('[man.web.chat]', 'hi', expect.any(Object))
  })

  it('child inherits context from parent', () => {
    const parent = new WebLogger('root', 'debug', { app: 'test' })
    const child = parent.child('sub', { extra: true })
    child.info('msg')
    const record = (console.log as ReturnType<typeof vi.fn>).mock.calls[0][2]
    expect(record.context.app).toBe('test')
    expect(record.context.extra).toBe(true)
  })

  it('getter/setter for level', () => {
    const log = new WebLogger('test', 'info')
    expect(log.level).toBe('info')
    log.level = 'debug'
    expect(log.level).toBe('debug')
  })

  it('toJSON serializes a record', () => {
    const log = new WebLogger('test')
    const record = { level: 'info' as const, logger: 'test', message: 'hi', timestamp: 1000 }
    const json = log.toJSON(record)
    expect(JSON.parse(json)).toEqual(record)
  })

  it('fromJSON deserializes a record', () => {
    const log = new WebLogger('test')
    const record = log.fromJSON('{"level":"info","logger":"test","message":"hi","timestamp":1000}')
    expect(record.message).toBe('hi')
    expect(record.level).toBe('info')
  })

  it('exports a singleton logger', () => {
    expect(logger.name).toBe('slo.web')
  })
})

describe('devDebug', () => {
  it('calls console.debug in development', () => {
    devDebug('test', 123)
    expect(console.debug).toHaveBeenCalledWith('[slo]', 'test', 123)
  })
})
