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

describe('persisted colours', () => {
  it('passes pre-token stored colours through untouched (migration retired 2026-10-24)', () => {
    const legacy: SessionGroup[] = [
      { id: 'g1', name: 'Work', color: '#8b5cf6', sessionIds: [], createdAt: 1 },
      { id: 'g2', name: 'Play', color: '#84cc16', sessionIds: ['s1'], createdAt: 2 },
    ]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(legacy))

    const { result } = renderHook(() => useSessionGroups([], vi.fn()))
    // `--group-N` slots are theme-constant and equal the retired hex values,
    // so stale storage renders the same hue until the user re-picks a colour.
    expect(result.current.groups[0].color).toBe('#8b5cf6')
    expect(result.current.groups[1].color).toBe('#84cc16')
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
