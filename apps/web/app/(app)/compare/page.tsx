'use client'
export const dynamic = 'force-dynamic'

import { PageContainer } from '@/components/PageContainer'
import { useComparison } from '@/hooks/useComparison'
import { ComparisonView, ComparisonHeader } from '@/components/compare/ComparisonView'

export default function ComparePage() {
  const {
    models,
    results,
    running,
    loading,
    snapshots,
    snapshotName,
    setSnapshotName,
    completedResults,
    bestMetrics,
    chartData,
    runBenchmark,
    runAll,
    clearResult,
    exportResults,
    saveSnapshot,
    loadSnapshot,
    deleteSnapshot,
  } = useComparison()

  return (
    <PageContainer
      title="Model Comparison"
      subtitle="Side-by-side benchmark results across models"
      headerRight={
        <ComparisonHeader
          completedResults={completedResults}
          snapshotName={snapshotName}
          onSnapshotNameChange={setSnapshotName}
          onSaveSnapshot={saveSnapshot}
          onExport={exportResults}
          onRunAll={runAll}
          loading={loading}
          running={running}
        />
      }
      loading={loading}
    >
      <ComparisonView
        models={models}
        loading={loading}
        results={results}
        running={running}
        snapshots={snapshots}
        snapshotName={snapshotName}
        onSnapshotNameChange={setSnapshotName}
        completedResults={completedResults}
        bestMetrics={bestMetrics}
        chartData={chartData}
        onBenchmark={runBenchmark}
        onClear={clearResult}
        onRunAll={runAll}
        onExport={exportResults}
        onSaveSnapshot={saveSnapshot}
        onLoadSnapshot={loadSnapshot}
        onDeleteSnapshot={deleteSnapshot}
      />
    </PageContainer>
  )
}
