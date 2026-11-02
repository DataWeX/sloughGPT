'use client'

/**
 * Personality — composition root.
 *
 * State, IO and persona management live in `usePersonalityWorkbench`;
 * rendering lives in the `personality-*` siblings next to this file.
 * Only the page shell belongs here: the loading/not-found guards and the
 * section order.
 */
import { PageContainer } from '@/components/PageContainer'
import { Skeleton } from '@sloughgpt/strui'
import { usePersonalityWorkbench } from './usePersonalityWorkbench'
import { PersonalityHeader } from './personality-header'
import { PersonalityPersonas } from './personality-personas'
import { PersonalityPresets } from './personality-presets'
import { PersonalityIdentity } from './personality-identity'
import { PersonalityVoiceStyle } from './personality-voice-style'
import { PersonalityEvolution } from './personality-evolution'

export default function PersonalityPage() {
  const wb = usePersonalityWorkbench()
  const { loading, profile } = wb

  if (loading) {
    return (
      <PageContainer title="Personality">
        <div className="space-y-4">
          <Skeleton className="h-32" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </PageContainer>
    )
  }

  if (!profile) return null

  return (
    <PageContainer title="Personality">
      <div className="space-y-4">
        <PersonalityHeader wb={wb} />
        <PersonalityPersonas wb={wb} />
        <PersonalityPresets wb={wb} />
        <PersonalityIdentity wb={wb} />
        <PersonalityVoiceStyle wb={wb} />
        <PersonalityEvolution wb={wb} />
      </div>
    </PageContainer>
  )
}
