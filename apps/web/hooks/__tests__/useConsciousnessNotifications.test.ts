// @vitest-environment jsdom
import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useConsciousnessNotifications } from '../useConsciousnessNotifications'

vi.mock('@/lib/consciousness-notifications', () => ({
  getNotifications: vi.fn().mockReturnValue([]),
  getUnreadCount: vi.fn().mockReturnValue(0),
  markAsRead: vi.fn(),
  clearNotifications: vi.fn(),
  addNotification: vi.fn(),
  onNotificationsChange: vi.fn().mockReturnValue(() => {}),
}))

import {
  getNotifications,
  getUnreadCount,
  markAsRead as markNotifAsRead,
  clearNotifications,
  addNotification as addNotif,
  onNotificationsChange,
} from '@/lib/consciousness-notifications'

const mockGetNotifications = vi.mocked(getNotifications)
const mockGetUnreadCount = vi.mocked(getUnreadCount)
const mockMarkAsRead = vi.mocked(markNotifAsRead)
const mockClearNotifications = vi.mocked(clearNotifications)
const mockAddNotification = vi.mocked(addNotif)
const mockOnNotificationsChange = vi.mocked(onNotificationsChange)

describe('useConsciousnessNotifications', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetNotifications.mockReturnValue([])
    mockGetUnreadCount.mockReturnValue(0)
    mockOnNotificationsChange.mockReturnValue(() => {})
  })

  it('Returns initial empty state', () => {
    const { result } = renderHook(() => useConsciousnessNotifications())
    expect(result.current.notifications).toEqual([])
    expect(result.current.unreadCount).toBe(0)
  })

  it('addNotification adds notification', () => {
    const { result } = renderHook(() => useConsciousnessNotifications())
    act(() => {
      result.current.addNotification('Title', 'Body', 'info')
    })
    expect(mockAddNotification).toHaveBeenCalledWith('Title', 'Body', 'info')
  })

  it('markAsRead marks notification as read', () => {
    const { result } = renderHook(() => useConsciousnessNotifications())
    act(() => {
      result.current.markAsRead('notif_123')
    })
    expect(mockMarkAsRead).toHaveBeenCalledWith('notif_123')
  })

  it('clearAll removes all notifications', () => {
    const { result } = renderHook(() => useConsciousnessNotifications())
    act(() => {
      result.current.clearAll()
    })
    expect(mockClearNotifications).toHaveBeenCalledTimes(1)
  })

  it('unreadCount updates reactively', () => {
    mockGetUnreadCount.mockReturnValue(5)
    const { result } = renderHook(() => useConsciousnessNotifications())
    expect(result.current.unreadCount).toBe(5)
  })

  it('Returns correct shape', () => {
    const { result } = renderHook(() => useConsciousnessNotifications())
    expect(result.current).toHaveProperty('notifications')
    expect(result.current).toHaveProperty('unreadCount')
    expect(result.current).toHaveProperty('markAsRead')
    expect(result.current).toHaveProperty('clearAll')
    expect(result.current).toHaveProperty('addNotification')
    expect(typeof result.current.markAsRead).toBe('function')
    expect(typeof result.current.clearAll).toBe('function')
    expect(typeof result.current.addNotification).toBe('function')
  })
})
