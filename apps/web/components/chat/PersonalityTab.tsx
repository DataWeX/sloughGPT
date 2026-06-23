'use client'

import { useState, useEffect } from 'react'
import { soulsController } from '@/lib/souls-controller'
import SoulVisualizer from '@/components/souls/SoulVisualizer'

interface PersonalityTabProps {
  soulName: string | null
}

export function PersonalityTab({ soulName }: PersonalityTabProps) {
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    soulsController.getTraitWeights()
      .then(w => {
        if (!cancelled && w && !('error' in w)) {
          setTraitWeights(w as Record<string, Record<string, number>>)
        }
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [soulName])

  if (loading) {
    return (
      <div className="space-y-2 animate-pulse">
        <div className="h-4 w-24 bg-muted rounded" />
        <div className="h-32 bg-muted rounded" />
      </div>
    )
  }

  if (!traitWeights) {
    return (
      <div className="text-[11px] text-muted-foreground py-4 text-center">
        No personality data available
      </div>
    )
  }

  return (
    <div>
      <SoulVisualizer traitWeights={traitWeights} currentSoulName={soulName} />
    </div>
  )
}
