/**
 * Tests for useToolProfile — fetches a single tool profile via listTools.
 */
// @vitest-environment jsdom
import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useToolProfile } from './use-tool-profile'
import { listTools, type ToolProfile } from '@/lib/tools-controller'

vi.mock('@/lib/tools-controller', () => ({
  listTools: vi.fn(),
}))

function makeProfile(id: string): ToolProfile {
  return {
    id,
    name: 'Writing Assistant',
    description: 'desc',
    icon: 'sparkle',
    params: [],
    options: { tone: [{ id: 'friendly', label: 'Friendly', description: '' }] },
    default_options: {},
    system_prompt: 'sp',
    max_tokens: 700,
  }
}

describe('useToolProfile', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('starts null and resolves to the matching profile', async () => {
    vi.mocked(listTools).mockResolvedValue([makeProfile('writing'), makeProfile('explain')])

    const { result } = renderHook(() => useToolProfile('writing'))

    expect(result.current).toBeNull()
    await waitFor(() => expect(result.current?.id).toBe('writing'))
    expect(listTools).toHaveBeenCalledTimes(1)
  })

  it('keeps null when the profile is not in the list', async () => {
    vi.mocked(listTools).mockResolvedValue([makeProfile('explain')])

    const { result } = renderHook(() => useToolProfile('writing'))

    await waitFor(() => expect(listTools).toHaveBeenCalledTimes(1))
    expect(result.current).toBeNull()
  })

  it('does not refetch on unrelated re-renders', async () => {
    vi.mocked(listTools).mockResolvedValue([makeProfile('writing')])

    const { result, rerender } = renderHook(() => useToolProfile('writing'))
    await waitFor(() => expect(result.current?.id).toBe('writing'))

    rerender()
    expect(listTools).toHaveBeenCalledTimes(1)
  })
})