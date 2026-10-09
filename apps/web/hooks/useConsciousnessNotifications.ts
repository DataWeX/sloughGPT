'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  getNotifications,
  getUnreadCount,
  markAsRead as markNotifAsRead,
  clearNotifications,
  addNotification as addNotif,
  onNotificationsChange,
  type ConsciousnessNotification,
  type ConsciousnessNotificationType,
} from '@/lib/consciousness-notifications'

export function useConsciousnessNotifications() {
  const [notifications, setNotifications] = useState<ConsciousnessNotification[]>([])
  const [unreadCount, setUnreadCount] = useState(0)

  const refresh = useCallback(() => {
    setNotifications(getNotifications())
    setUnreadCount(getUnreadCount())
  }, [])

  useEffect(() => {
    refresh()
    const unsub = onNotificationsChange(refresh)
    return unsub
  }, [refresh])

  const markAsRead = useCallback((id: string) => {
    markNotifAsRead(id)
  }, [])

  const clearAll = useCallback(() => {
    clearNotifications()
  }, [])

  const addNotification = useCallback((title: string, body: string, type: ConsciousnessNotificationType) => {
    addNotif(title, body, type)
  }, [])

  return { notifications, unreadCount, markAsRead, clearAll, addNotification }
}
