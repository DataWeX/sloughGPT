'use client'

// Consciousness dashboard — thin composition. All state, fetching, and action
// logic lives in ./useConsciousnessDashboard; section markup lives in
// ./components/* (extracted by the page-decomposition refactor).

import { PageContainer } from '@/components/PageContainer'
import { useLocale } from '@/hooks/useLocale'
import { useConsciousnessDashboard } from './useConsciousnessDashboard'
import { DashboardActions, DashboardHero } from './components/DashboardHero'
import { DashboardSummary } from './components/DashboardSummary'
import { DashboardCharts } from './components/DashboardCharts'
import { EpisodesTimeline } from './components/EpisodesTimeline'
import { EpisodeDetailDialog } from './components/EpisodeDetailDialog'
import { ConsciousnessLevelFold, CurrentStateFold } from './components/DashboardFolds'

export default function ConsciousnessDashboardPage() {
  const { t } = useLocale()
  const d = useConsciousnessDashboard()

  if (d.loading) {
    return <PageContainer title="Consciousness Dashboard" loadingGrid />
  }

  return (
    <PageContainer
      title={
        <DashboardHero
          isLive={d.isLive}
          lastUpdate={d.lastUpdate}
          toggleLive={d.toggleLive}
          liveLabel={
            d.isLive ? t('consciousness_dashboard.live') : t('consciousness_dashboard.autoRefresh')
          }
        />
      }
      headerRight={
        <DashboardActions
          seeding={d.seeding}
          reflecting={d.reflecting}
          exporting={d.exporting}
          importing={d.importing}
          onSeed={d.handleSeed}
          onReflect={d.handleReflect}
          onExport={d.handleExport}
          onImport={d.handleImport}
        />
      }
    >
      <DashboardSummary
        totalEpisodes={d.totalEpisodes}
        score={d.score}
        avgGrowth={d.avgGrowth}
        consciousnessLevel={d.consciousnessLevel}
        isLive={d.isLive}
        evalReport={d.evalReport}
        lastReflection={d.lastReflection}
      />
      <DashboardCharts
        beliefsData={d.beliefsData}
        qualiaData={d.qualiaData}
        growthData={d.growthData}
      />
      <EpisodesTimeline
        episodes={d.episodes}
        ratingEpisode={d.ratingEpisode}
        onRate={d.handleFeedback}
        onSelect={d.setSelectedEpisode}
      />
      <ConsciousnessLevelFold level={d.consciousnessLevel} onLevelChange={d.handleLevelChange} />
      <CurrentStateFold currentBeliefs={d.currentBeliefs} currentQualia={d.currentQualia} />
      <EpisodeDetailDialog
        episode={d.selectedEpisode}
        episodes={d.episodes}
        onRate={d.handleFeedback}
        onClose={() => d.setSelectedEpisode(null)}
      />
    </PageContainer>
  )
}
