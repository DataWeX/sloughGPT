'use client'

/**
 * Consciousness Dashboard — composition root.
 *
 * State and IO live in `useConsciousnessDashboard`; rendering lives in the
 * `dashboard-*` siblings. This file only wires the two together, so adding a
 * section means adding a panel, not growing this page.
 */
import { PageContainer } from '@/components/PageContainer'
import { useConsciousnessDashboard } from '@/hooks/useConsciousnessDashboard'
import { DashboardActions, DashboardHeadline } from './dashboard-header'
import { DashboardMetrics } from './dashboard-metrics'
import { DashboardEvolution } from './dashboard-evolution'
import { DashboardEpisodes, EpisodeDetailDialog } from './dashboard-episodes'
import { DashboardState } from './dashboard-state'

export default function ConsciousnessDashboardPage() {
  const {
    loading,
    episodes,
    totalEpisodes,
    qualiaHistory,
    beliefsHistory,
    evalReport,
    currentBeliefs,
    currentQualia,
    consciousnessLevel,
    isLive,
    lastUpdate,
    toggleLive,
    seeding,
    reflecting,
    ratingEpisode,
    exporting,
    importing,
    selectedEpisode,
    setSelectedEpisode,
    closeSelectedEpisode,
    handleSeed,
    handleReflect,
    handleFeedback,
    handleLevelChange,
    handleExport,
    handleImport,
  } = useConsciousnessDashboard()

  if (loading) {
    return <PageContainer title="Consciousness Dashboard" loadingGrid />
  }

  return (
    <PageContainer
      title={
        <DashboardHeadline
          isLive={isLive}
          lastUpdate={lastUpdate}
          onToggleLive={toggleLive}
        />
      }
      headerRight={
        <DashboardActions
          seeding={seeding}
          reflecting={reflecting}
          exporting={exporting}
          importing={importing}
          onSeed={handleSeed}
          onReflect={handleReflect}
          onExport={handleExport}
          onImport={handleImport}
        />
      }
    >
      <DashboardMetrics
        totalEpisodes={totalEpisodes}
        episodes={episodes}
        evalReport={evalReport}
        consciousnessLevel={consciousnessLevel}
        isLive={isLive}
      />

      <DashboardEvolution
        qualiaHistory={qualiaHistory}
        beliefsHistory={beliefsHistory}
        episodes={episodes}
      />

      <DashboardEpisodes
        episodes={episodes}
        ratingEpisode={ratingEpisode}
        onSelectEpisode={setSelectedEpisode}
        onRate={handleFeedback}
      />

      <DashboardState
        consciousnessLevel={consciousnessLevel}
        currentBeliefs={currentBeliefs}
        currentQualia={currentQualia}
        onLevelChange={handleLevelChange}
      />

      <EpisodeDetailDialog
        episode={selectedEpisode}
        episodes={episodes}
        onClose={closeSelectedEpisode}
        onRate={handleFeedback}
      />
    </PageContainer>
  )
}
