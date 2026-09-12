'use client'

export type ConsciousnessNotificationType = 'info' | 'warning' | 'success' | 'error'

export interface ConsciousnessNotification {
  id: string
  title: string
  body: string
  type: ConsciousnessNotificationType
  timestamp: number
  read: boolean
}

const STORAGE_KEY = 'consciousness_notifications'
const MAX_NOTIFICATIONS = 100

type Listener = () => void
const listeners = new Set<Listener>()

function notifyListeners() {
  for (const fn of listeners) fn()
}

function load(): ConsciousnessNotification[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw) as ConsciousnessNotification[]
  } catch {
    return []
  }
}

function save(notifications: ConsciousnessNotification[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(notifications))
}

export function addNotification(title: string, body: string, type: ConsciousnessNotificationType): ConsciousnessNotification {
  const notification: ConsciousnessNotification = {
    id: `notif_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    title,
    body,
    type,
    timestamp: Date.now(),
    read: false,
  }
  const all = load()
  all.unshift(notification)
  if (all.length > MAX_NOTIFICATIONS) all.length = MAX_NOTIFICATIONS
  save(all)
  notifyListeners()
  return notification
}

export function getNotifications(): ConsciousnessNotification[] {
  return load()
}

export function getUnreadCount(): number {
  return load().filter(n => !n.read).length
}

export function markAsRead(id: string) {
  const all = load()
  const notif = all.find(n => n.id === id)
  if (notif) {
    notif.read = true
    save(all)
    notifyListeners()
  }
}

export function clearNotifications() {
  save([])
  notifyListeners()
}

export function onNotificationsChange(fn: Listener): () => void {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}
