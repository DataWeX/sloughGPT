// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { COLORS, useSessionGroups, type SessionGroup } from './useSessionGroups'
import { renderHook, act } from '@testing-library/react'

const STORAGE_KEY = 'chat-session-groups'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('COLORS', () => {
  it('is ten theme-constant token references, no raw hues', () => {
    expect(COLORS).toHaveLength(10)
    for (const [i, c] of COLORS.entries()) {
      expect(c).toBe(`var(--group-${i + 1})`)
    }
    expect(COLORS.join(' ')).not.toMatch(/#[0-9a-fA-F]{3,8}/)
  })
})

describe('legacy hex migration', () => {
  it('maps persisted hex colours to group tokens on load', () => {
    const legacy: SessionGroup[] = [
      { id: 'g1', name: 'Work', color: '#8b5cf6', sessionIds: [], createdAt: 1 },
      { id: 'g2', name: 'Play', color: '#84cc16', sessionIds: ['s1'], createdAt: 2 },
    ]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(legacy))

    const { result } = renderHook(() => useSessionGroups([], vi.fn()))
    expect(result.current.groups[0].color).toBe('var(--group-1)')
    expect(result.current.groups[1].color).toBe('var(--group-10)')
  })

  it('round-trips token references unchanged', () => {
    const tokenized: SessionGroup[] = [
      { id: 'g1', name: 'Work', color: 'var(--group-3)', sessionIds: [], createdAt: 1 },
    ]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tokenized))

    const { result } = renderHook(() => useSessionGroups([], vi.fn()))
    expect(result.current.groups[0].color).toBe('var(--group-3)')
  })

  it('new groups default to the first group token', () => {
    const { result } = renderHook(() => useSessionGroups([], vi.fn()))
    expect(result.current.newColor).toBe('var(--group-1)')

    act(() => {
      result.current.setNewName('Design')
    })
    act(() => {
      result.current.handleCreate()
    })
    expect(result.current.groups[0].color).toBe('var(--group-1)')
  })
})
