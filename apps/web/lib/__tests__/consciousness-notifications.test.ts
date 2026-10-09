import { describe, it, expect, vi, beforeEach } from 'vitest'

const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value }),
    removeItem: vi.fn((key: string) => { delete store[key] }),
    clear: vi.fn(() => { store = {} }),
    get store() { return store },
    reset() { store = {} },
  }
})()
vi.stubGlobal('localStorage', localStorageMock)
vi.stubGlobal('window', { localStorage: localStorageMock })

import {
  addNotification,
  getNotifications,
  getUnreadCount,
  markAsRead,
  clearNotifications,
  onNotificationsChange,
} from '../consciousness-notifications'

describe('consciousness-notifications', () => {
  beforeEach(() => {
    localStorageMock.reset()
    localStorageMock.getItem.mockClear()
    localStorageMock.setItem.mockClear()
  })

  it('getNotifications returns empty array initially', () => {
    const result = getNotifications()
    expect(result).toEqual([])
  })

  it('addNotification returns notification with id', () => {
    const notif = addNotification('Test Title', 'Test Body', 'info')
    expect(notif).toHaveProperty('id')
    expect(typeof notif.id).toBe('string')
    expect(notif.id.length).toBeGreaterThan(0)
  })

  it('addNotification stores notification', () => {
    addNotification('Test Title', 'Test Body', 'warning')
    const stored = getNotifications()
    expect(stored).toHaveLength(1)
    expect(stored[0].title).toBe('Test Title')
    expect(stored[0].body).toBe('Test Body')
    expect(stored[0].type).toBe('warning')
    expect(stored[0].read).toBe(false)
  })

  it('addNotification notifies listeners', () => {
    const listener = vi.fn()
    onNotificationsChange(listener)
    addNotification('Title', 'Body', 'success')
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('getNotifications returns stored notifications', () => {
    addNotification('Title 1', 'Body 1', 'info')
    addNotification('Title 2', 'Body 2', 'error')
    const result = getNotifications()
    expect(result).toHaveLength(2)
    expect(result[0].title).toBe('Title 2')
    expect(result[1].title).toBe('Title 1')
  })

  it('getUnreadCount returns correct count', () => {
    addNotification('Title 1', 'Body 1', 'info')
    addNotification('Title 2', 'Body 2', 'error')
    expect(getUnreadCount()).toBe(2)
    const first = getNotifications()[1]
    markAsRead(first.id)
    expect(getUnreadCount()).toBe(1)
  })

  it('markAsRead marks notification as read', () => {
    const notif = addNotification('Title', 'Body', 'info')
    expect(getNotifications()[0].read).toBe(false)
    markAsRead(notif.id)
    expect(getNotifications()[0].read).toBe(true)
  })

  it('markAsRead notifies listeners', () => {
    const notif = addNotification('Title', 'Body', 'info')
    const listener = vi.fn()
    onNotificationsChange(listener)
    markAsRead(notif.id)
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('clearNotifications clears all', () => {
    addNotification('Title 1', 'Body 1', 'info')
    addNotification('Title 2', 'Body 2', 'error')
    expect(getNotifications()).toHaveLength(2)
    clearNotifications()
    expect(getNotifications()).toHaveLength(0)
  })

  it('clearNotifications notifies listeners', () => {
    const listener = vi.fn()
    onNotificationsChange(listener)
    clearNotifications()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('onNotificationsChange returns unsubscribe function', () => {
    const listener = vi.fn()
    const unsub = onNotificationsChange(listener)
    addNotification('Title', 'Body', 'info')
    expect(listener).toHaveBeenCalledTimes(1)
    unsub()
    addNotification('Title 2', 'Body 2', 'error')
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('Multiple notifications ordered newest first', () => {
    addNotification('First', 'Body 1', 'info')
    addNotification('Second', 'Body 2', 'warning')
    addNotification('Third', 'Body 3', 'error')
    const result = getNotifications()
    expect(result).toHaveLength(3)
    expect(result[0].title).toBe('Third')
    expect(result[1].title).toBe('Second')
    expect(result[2].title).toBe('First')
  })
})
