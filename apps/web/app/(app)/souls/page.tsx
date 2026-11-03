'use client'

/**
 * Souls — composition root.
 *
 * State and IO live in `useSoulsWorkbench`; rendering lives in the
 * `souls-*` siblings next to this file. Only the page shell belongs here:
 * the loading skeleton, the KPI strip, the tab switch (which lazy-loads
 * weights/snapshots on first entry) and the dialog wiring.
 */
import { PageContainer } from '@/components/PageContainer'
import {
  Card,
  CardContent,
  KpiGrid,
  Skeleton,
  StatCard,
  Button,
  Tabs,
  TabsList,
  TabsTrigger,
} from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { useSoulsWorkbench, type Tab } from './useSoulsWorkbench'
import { SoulsListTab } from './souls-list-tab'
import { CheckpointsTab } from './souls-checkpoints-tab'
import { WeightsTab } from './souls-weights-tab'
import { SnapshotsTab } from './souls-snapshots-tab'
import { AnalyticsTab } from './souls-analytics-tab'
import { SoulDetailDialog } from './souls-detail-dialog'
import { SoulCompareDialog } from './souls-compare-dialog'
import { CheckpointDetailDialog } from './souls-checkpoint-dialog'
import { DeleteConfirmDialog } from './souls-delete-dialog'

export default function SoulsPage() {
  const wb = useSoulsWorkbench()
  const { loading, souls, currentSoul, checkpoints, snapshots, tab, setTab, handleRefresh } = wb

  // ── Loading State ──
  if (loading) {
    return (
      <PageContainer
        title="Souls"
        subtitle="Personality management"
        loading
        loadingContent={
          <div className="space-y-4">
            <KpiGrid>
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
              <StatCard label="Loading" value={<Skeleton className="h-5 w-12" />} />
            </KpiGrid>
            <Card>
              <CardContent>
                <Skeleton className="h-32 w-full rounded" />
              </CardContent>
            </Card>
          </div>
        }
      >
        <></>
      </PageContainer>
    )
  }

  // ── Render ──
  return (
    <PageContainer
      title="Souls"
      subtitle={`${souls.length} personalities · ${currentSoul ?? 'none'} active`}
      headerRight={
        <Button size="sm" variant="ghost" onClick={handleRefresh} aria-label="Refresh">
          <IconRefresh className="h-4 w-4" />
        </Button>
      }
    >
      <KpiGrid>
        <StatCard label="Personalities" value={souls.length} />
        <StatCard label="Active Soul" value={currentSoul ?? 'None'} />
        <StatCard label="Checkpoints" value={checkpoints.length} />
        <StatCard label="Snapshots" value={snapshots.length || '—'} />
      </KpiGrid>

      <Tabs
        value={tab}
        onValueChange={(v) => {
          setTab(v as Tab)
          if (v === 'weights') wb.handleLoadWeights()
          if (v === 'snapshots') wb.handleLoadSnapshots()
        }}
      >
        <TabsList>
          <TabsTrigger value="souls">Souls</TabsTrigger>
          <TabsTrigger value="checkpoints">Checkpoints</TabsTrigger>
          <TabsTrigger value="weights">Weights</TabsTrigger>
          <TabsTrigger value="snapshots">Snapshots</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* Each tab panel renders its own <TabsContent> wrapper. */}
        <SoulsListTab wb={wb} />
        <CheckpointsTab wb={wb} />
        <WeightsTab wb={wb} />
        <SnapshotsTab wb={wb} />
        <AnalyticsTab wb={wb} />
      </Tabs>

      {/* ═══════════════ DIALOGS ═══════════════ */}
      <SoulDetailDialog wb={wb} />
      <SoulCompareDialog wb={wb} />
      <CheckpointDetailDialog wb={wb} />
      <DeleteConfirmDialog wb={wb} />
    </PageContainer>
  )
}
