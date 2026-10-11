'use client'

import { PageContainer } from '@/components/PageContainer'
import { DashboardActions } from '@/components/consciousness-dashboard/DashboardActions'
import { DashboardControls } from '@/components/consciousness-dashboard/DashboardControls'
import { DashboardEpisodeTimeline } from '@/components/consciousness-dashboard/DashboardEpisodeTimeline'
import { DashboardEvolution } from '@/components/consciousness-dashboard/DashboardEvolution'
import { DashboardFocalMetrics } from '@/components/consciousness-dashboard/DashboardFocalMetrics'
import { DashboardHeader } from '@/components/consciousness-dashboard/DashboardHeader'
import { DashboardResponseQuality } from '@/components/consciousness-dashboard/DashboardResponseQuality'
import { EpisodeDetailModal } from '@/components/consciousness-dashboard/EpisodeDetailModal'
import { useConsciousnessDashboard } from '@/hooks/useConsciousnessDashboard'

/**
 * Consciousness Dashboard.
 *
 * All data fetching and state live in `useConsciousnessDashboard`; each visual
 * section is a component under `components/consciousness-dashboard`. This file
 * is intentionally only composition, so adding a section means adding one
 * component and one line here.
 */
export default function ConsciousnessDashboardPage() {
  const d = useConsciousnessDashboard()

  if (d.loading) {
    return <PageContainer title="Consciousness Dashboard" loadingGrid />
  }

  return (
    <PageContainer
      title={
        <DashboardHeader isLive={d.isLive} lastUpdate={d.lastUpdate} onToggleLive={d.toggleLive} />
      }
      headerRight={
        <DashboardActions
          onSeed={d.handleSeed}
          seeding={d.seeding}
          onReflect={d.handleReflect}
          reflecting={d.reflecting}
          onExport={d.handleExport}
          exporting={d.exporting}
          onImport={d.handleImport}
          importing={d.importing}
        />
      }
    >
      <DashboardFocalMetrics
        totalEpisodes={d.totalEpisodes}
        score={d.score}
        avgGrowth={d.avgGrowth}
        consciousnessLevel={d.consciousnessLevel}
        isLive={d.isLive}
      />

      <DashboardResponseQuality score={d.score} evalReport={d.evalReport} episodes={d.episodes} />

      <DashboardEvolution
        beliefsData={d.beliefsData}
        qualiaData={d.qualiaData}
        growthData={d.growthData}
      />

      <DashboardEpisodeTimeline
        episodes={d.episodes}
        ratingEpisode={d.ratingEpisode}
        onFeedback={d.handleFeedback}
        onSelect={d.setSelectedEpisode}
      />

      <DashboardControls
        consciousnessLevel={d.consciousnessLevel}
        onLevelChange={d.handleLevelChange}
        currentBeliefs={d.currentBeliefs}
        currentQualia={d.currentQualia}
      />

      <EpisodeDetailModal
        episode={d.selectedEpisode}
        episodes={d.episodes}
        onRate={d.handleFeedback}
        onClose={() => d.setSelectedEpisode(null)}
      />
    </PageContainer>
  )
}
