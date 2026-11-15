'use client'

/**
 * Personality tab — folded from the old /personality page (route-map card 4/7).
 *
 * State, IO and persona management live in `usePersonalityWorkbench`;
 * rendering lives in the `personality-*` siblings next to this file.
 * The workbench mounts with the panel — Radix unmounts inactive
 * TabsContent, so the profile fetch fires on first tab entry (same
 * lazy contract as the weights/snapshots tabs).
 */
import { Skeleton, TabsContent } from '@sloughgpt/strui'
import { usePersonalityWorkbench } from './usePersonalityWorkbench'
import { PersonalityHeader } from './personality-header'
import { PersonalityPersonas } from './personality-personas'
import { PersonalityPresets } from './personality-presets'
import { PersonalityIdentity } from './personality-identity'
import { PersonalityVoiceStyle } from './personality-voice-style'
import { PersonalityEvolution } from './personality-evolution'

function PersonalityPanel() {
  const wb = usePersonalityWorkbench()
  const { loading, profile } = wb

  if (loading) {
    return (
      <div className="space-y-4" data-testid="personality-loading">
        <Skeleton className="h-32" />
        <Skeleton className="h-48" />
        <Skeleton className="h-48" />
      </div>
    )
  }

  if (!profile) return null

  return (
    <div className="space-y-4">
      <PersonalityHeader wb={wb} />
      <PersonalityPersonas wb={wb} />
      <PersonalityPresets wb={wb} />
      <PersonalityIdentity wb={wb} />
      <PersonalityVoiceStyle wb={wb} />
      <PersonalityEvolution wb={wb} />
    </div>
  )
}

export function PersonalityTab() {
  return (
    <TabsContent value="personality" className="space-y-4">
      <PersonalityPanel />
    </TabsContent>
  )
}
