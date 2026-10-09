/**
 * useToolProfile — loads a single tool profile from the backend GET /tools
 * metadata. Returns null while loading or when the tool id is unknown.
 * Pages use this as the source of truth for option groups, falling back to
 * static lists when the backend is unreachable.
 */

'use client'

import { useEffect, useState } from 'react'
import { listTools, type ToolProfile } from '@/lib/tools-controller'

export function useToolProfile(toolId: string): ToolProfile | null {
  const [profile, setProfile] = useState<ToolProfile | null>(null)

  useEffect(() => {
    let active = true
    setProfile(null)
    listTools().then((tools) => {
      if (!active) return
      setProfile(tools.find((t) => t.id === toolId) ?? null)
    })
    return () => {
      active = false
    }
  }, [toolId])

  return profile
}