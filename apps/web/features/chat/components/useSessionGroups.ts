'use client'

import { useState, useCallback, useMemo, useEffect } from 'react'

export interface SessionGroup {
  id: string
  name: string
  color: string
  sessionIds: string[]
  createdAt: number
}

export interface Session {
  id: string
  title: string
}

/**
 * Group identity uses the theme-constant `--group-N` slots (globals.css) —
 * markup never pins a hue, so the picker and dots follow the token table.
 */
export const COLORS = Array.from({ length: 10 }, (_, i) => `var(--group-${i + 1})`)

/**
 * Persisted groups from before the token migration store raw hex. Map them
 * to their slot once, in order, so existing groups keep their hue.
 */
const LEGACY_HEX_TO_TOKEN: Record<string, string> = {
  '#8b5cf6': 'var(--group-1)',
  '#3b82f6': 'var(--group-2)',
  '#10b981': 'var(--group-3)',
  '#f59e0b': 'var(--group-4)',
  '#ef4444': 'var(--group-5)',
  '#ec4899': 'var(--group-6)',
  '#6366f1': 'var(--group-7)',
  '#14b8a6': 'var(--group-8)',
  '#f97316': 'var(--group-9)',
  '#84cc16': 'var(--group-10)',
}

function migrateColor(color: string): string {
  return LEGACY_HEX_TO_TOKEN[color.toLowerCase()] ?? color
}

const STORAGE_KEY = 'chat-session-groups'

function loadGroups(): SessionGroup[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map((g: SessionGroup) => ({ ...g, color: migrateColor(g.color) }))
  } catch {
    return []
  }
}

function saveGroups(groups: SessionGroup[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(groups))
}

export interface UseSessionGroupsReturn {
  groups: SessionGroup[]
  creating: boolean
  newName: string
  newColor: string
  selectedColor: string | null
  assigningTo: string | null
  sessionsByGroup: { map: Record<string, Session[]>; ungrouped: Session[] }
  setCreating: (v: boolean) => void
  setNewName: (v: string) => void
  setNewColor: (v: string) => void
  setSelectedColor: (v: string | null) => void
  setAssigningTo: (v: string | null) => void
  handleCreate: () => void
  handleDelete: (id: string) => void
  handleAssign: (sessionId: string, groupId: string) => void
  handleUnassign: (sessionId: string) => void
}

export function useSessionGroups(
  sessions: Session[],
  onAssignGroup: (sessionId: string, groupId: string | null) => void,
): UseSessionGroupsReturn {
  const [groups, setGroups] = useState<SessionGroup[]>([])
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newColor, setNewColor] = useState(COLORS[0])
  const [selectedColor, setSelectedColor] = useState<string | null>(null)
  const [assigningTo, setAssigningTo] = useState<string | null>(null)

  useEffect(() => {
    setGroups(loadGroups())
  }, [])

  useEffect(() => {
    saveGroups(groups)
  }, [groups])

  const sessionsByGroup = useMemo(() => {
    const map: Record<string, Session[]> = {}
    const ungrouped: Session[] = []

    for (const session of sessions) {
      const group = groups.find((g) => g.sessionIds.includes(session.id))
      if (group) {
        if (!map[group.id]) map[group.id] = []
        map[group.id].push(session)
      } else {
        ungrouped.push(session)
      }
    }

    return { map, ungrouped }
  }, [groups, sessions])

  const handleCreate = useCallback(() => {
    const trimmed = newName.trim()
    if (!trimmed) return

    const newGroup: SessionGroup = {
      id: `group-${Date.now()}`,
      name: trimmed,
      color: newColor,
      sessionIds: [],
      createdAt: Date.now(),
    }

    const next = [...groups, newGroup]
    setGroups(next)
    setNewName('')
    setCreating(false)
  }, [newName, newColor, groups])

  const handleDelete = useCallback(
    (id: string) => {
      const next = groups.filter((g) => g.id !== id)
      setGroups(next)
    },
    [groups],
  )

  const handleAssign = useCallback(
    (sessionId: string, groupId: string) => {
      const next = groups.map((g) => {
        if (g.id === groupId) {
          return { ...g, sessionIds: [...g.sessionIds, sessionId] }
        }
        return { ...g, sessionIds: g.sessionIds.filter((id) => id !== sessionId) }
      })
      setGroups(next)
      onAssignGroup(sessionId, groupId)
      setAssigningTo(null)
    },
    [groups, onAssignGroup],
  )

  const handleUnassign = useCallback(
    (sessionId: string) => {
      const next = groups.map((g) => ({
        ...g,
        sessionIds: g.sessionIds.filter((id) => id !== sessionId),
      }))
      setGroups(next)
      onAssignGroup(sessionId, null)
      setAssigningTo(null)
    },
    [groups, onAssignGroup],
  )

  return {
    groups,
    creating,
    newName,
    newColor,
    selectedColor,
    assigningTo,
    sessionsByGroup,
    setCreating,
    setNewName,
    setNewColor,
    setSelectedColor,
    setAssigningTo,
    handleCreate,
    handleDelete,
    handleAssign,
    handleUnassign,
  }
}
