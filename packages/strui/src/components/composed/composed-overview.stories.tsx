import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { IconDownload, IconRefresh, IconSettings } from '../ui/icons'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Switch } from '../ui/switch'
import { AppShell } from './app-shell'
import { EmptyCard } from './empty-card'
import { FormField } from './form-field'
import { InlineBanner } from './inline-banner'
import { KpiGrid } from './kpi-grid'
import { NavRail, NavRailLink } from './nav-rail'
import { PageHeader } from './page-header'
import { SearchInput } from './search-input'
import { SettingsRow } from './settings-row'
import { Skeleton } from './skeleton'
import { StatCard } from './stat-card'
import { Toolbar } from './toolbar'
import { ActionCard } from './action-card'
import { ChipGroup } from './chip-group'
import { InsightsCard } from './insights-card'
import { MetricsCard } from './metrics-card'
import { StatusBadge } from './status-badge'
import { Chip } from './chip'
import { Breadcrumbs } from './breadcrumbs'
import { FoldSection } from './fold-section'
import { Kbd } from './kbd'
import { KeyValueList } from './key-value-list'
import { ListRow } from './list-row'
import { ProgressBar } from './progress-bar'
import { ScrollPanel } from './scroll-panel'
import { SectionHeader } from './section-header'
import { StatusDot } from './status-dot'
import { StepIndicator } from './step-indicator'
import { Timeline } from './timeline'

const meta = {
  title: 'Composed/Overview',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story:
          'Complete strui composed component library: shell patterns, dashboards, cards, metadata, progress, and all composite building blocks.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta

export default meta

// ── Shell & Layout ────────────────────────────────────────────────

export const DashboardShell: StoryObj = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <AppShell
        sidebar={
          <NavRail
            header={<span className="px-3 text-xs font-bold uppercase tracking-wider">Acme</span>}
          >
            <NavRailLink href="#" active>
              Home
            </NavRailLink>
            <NavRailLink href="#">Agents</NavRailLink>
            <NavRailLink href="#">Settings</NavRailLink>
          </NavRail>
        }
      >
        <PageHeader
          title="Overview"
          description="Composed building blocks for dashboards and AI apps."
          actions={
            <>
              <Button size="sm">New</Button>
              <Button size="sm" variant="secondary">
                Export
              </Button>
            </>
          }
        />
        <Toolbar>
          <SearchInput className="max-w-sm" placeholder="Filter…" />
          <Button size="sm" variant="ghost">
            Refresh
          </Button>
        </Toolbar>
        <div className="str-safe-x space-y-6 p-4">
          <InlineBanner
            variant="warning"
            title="Rate limit"
            description="You are approaching the tier cap for this workspace."
            action={<Button size="sm">Upgrade</Button>}
          />
          <KpiGrid>
            <StatCard
              label="Requests"
              value="12.4k"
              icon="📈"
              trend={{ value: 3, positive: true }}
            />
            <StatCard label="Latency p95" value="420ms" />
            <StatCard label="Errors" value="0.02%" />
            <StatCard label="Cost" value="$48" />
          </KpiGrid>
          <div className="grid gap-6 lg:grid-cols-2">
            <EmptyCard
              message="No experiments"
              action={
                <Button className="str-touch-target w-full sm:w-auto">Create experiment</Button>
              }
            />
            <div className="space-y-3">
              <p className="text-xs font-medium uppercase text-muted-foreground">
                Loading skeleton
              </p>
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-[75%]" />
              <Skeleton className="h-24 w-full" />
            </div>
          </div>
          <SettingsRow
            title="Enable streaming"
            description="Show tokens as they are generated."
            control={<Switch checked={on} onCheckedChange={setOn} />}
          />
          <FormField id="api" label="API key" hint="Stored in the browser for this demo.">
            <Input id="api" placeholder="sk-…" autoComplete="off" />
          </FormField>
        </div>
      </AppShell>
    )
  },
}

// ── Full Dashboard Layout ─────────────────────────────────────────

export const FullDashboard: StoryObj = {
  render: () => (
    <div className="max-w-4xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Model Training</h1>
          <p className="text-xs text-muted-foreground">Manage and monitor your training jobs</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm">
            <IconDownload className="h-3 w-3 mr-1" />
            New Job
          </Button>
        </div>
      </div>

      <MetricsCard title="Overview" columns={4}>
        <StatCard label="Total Jobs" value={42} />
        <StatCard label="Active" value={3} trend={{ value: 12, positive: true }} />
        <StatCard label="Failed" value={1} trend={{ value: 5, positive: false }} />
        <StatCard label="Avg Time" value="2.4h" />
      </MetricsCard>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InsightsCard
          title="Training Insights"
          kpis={[
            { label: 'Loss', value: '0.23' },
            { label: 'Accuracy', value: '94%' },
          ]}
          details={[
            { label: 'Epochs completed', value: 12 },
            { label: 'Learning rate', value: '2e-4' },
            { label: 'Batch size', value: 32 },
          ]}
        />
      </div>
    </div>
  ),
}

// ── ActionCard ─────────────────────────────────────────────────────

export const ActionCardDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4">
      <ActionCard
        title="System Health"
        subtitle="Last updated 2m ago"
        actions={
          <>
            <Button size="sm" variant="outline">
              <IconRefresh className="h-3 w-3" />
            </Button>
            <Button size="sm" variant="ghost">
              <IconSettings className="h-3 w-3" />
            </Button>
          </>
        }
      >
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">CPU</span>
            <span>45%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Memory</span>
            <span>8.2 GB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Disk</span>
            <span>120 GB</span>
          </div>
        </div>
      </ActionCard>
    </div>
  ),
}

// ── MetricsCard ────────────────────────────────────────────────────

export const MetricsCardDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4">
      <MetricsCard title="Resources" columns={2}>
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">CPU</p>
          <p className="text-xl font-semibold mt-1">45%</p>
        </div>
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Memory</p>
          <p className="text-xl font-semibold mt-1">8.2 GB</p>
        </div>
      </MetricsCard>
    </div>
  ),
}

// ── InsightsCard ───────────────────────────────────────────────────

export const InsightsCardDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4">
      <InsightsCard
        title="Feedback Insights"
        kpis={[
          { label: 'Sentiment', value: '85%' },
          { label: 'Quality', value: 'Excellent' },
        ]}
        details={[
          { label: 'Total conversations', value: 42 },
          { label: 'Total messages', value: 128 },
          { label: 'Avg response time', value: '2.3s' },
        ]}
      />
    </div>
  ),
}

// ── ChipGroup ──────────────────────────────────────────────────────

export const ChipGroupDemo: StoryObj = {
  render: () => (
    <div className="p-4 space-y-4">
      <ChipGroup
        chips={[
          { label: 'Python', tone: 'primary' },
          { label: 'PyTorch', tone: 'success' },
          { label: 'GPU', tone: 'warning' },
          { label: 'Training', tone: 'default' },
        ]}
      />
      <ChipGroup
        chips={[{ label: 'Filter: active' }, { label: 'Status: running', tone: 'success' }]}
        gap={2}
      />
    </div>
  ),
}

// ── StatusBadge ────────────────────────────────────────────────────

export const StatusBadgeDemo: StoryObj = {
  render: () => (
    <div className="p-4 flex flex-wrap gap-3">
      <StatusBadge tone="success">Running</StatusBadge>
      <StatusBadge tone="warning">Queued</StatusBadge>
      <StatusBadge tone="destructive">Failed</StatusBadge>
      <StatusBadge tone="primary">Active</StatusBadge>
      <StatusBadge tone="muted">Idle</StatusBadge>
    </div>
  ),
}

// ── Metadata & Progress ───────────────────────────────────────────

export const MetadataAndProgress: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-2xl space-y-8 p-4">
      <Breadcrumbs
        className="text-[0.7rem]"
        items={[{ label: 'Training', href: '#' }, { label: 'Job draft' }]}
      />
      <SectionHeader
        title="Run configuration"
        description="Read-only snapshot for this job."
        action={<Chip variant="primary">v2</Chip>}
      />
      <p className="text-xs text-muted-foreground">
        Tip: <Kbd>⌘</Kbd>
        <Kbd>K</Kbd> command palette (host app)
      </p>
      <KeyValueList
        items={[
          { label: 'Run id', value: 'run_01a2b3c', mono: true },
          { label: 'Dataset', value: 'sft-v1.jsonl' },
          { label: 'Learning rate', value: '2e-4' },
        ]}
      />
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase text-muted-foreground">Progress</p>
        <ProgressBar value={62} />
        <ProgressBar indeterminate className="opacity-80" />
      </div>
      <StepIndicator steps={['Prepare', 'Train', 'Export'] as const} current={1} />
      <ScrollPanel className="p-3 font-mono text-xs">
        {Array.from({ length: 12 }, (_, i) => (
          <div key={i} className="py-0.5 text-muted-foreground">
            [{String(i).padStart(2, '0')}] log line…
          </div>
        ))}
      </ScrollPanel>
      <div className="divide-y divide-border rounded-none border border-border">
        <ListRow label="Experiment A" value="Last updated 2h ago" action="›" />
        <ListRow label="Experiment B" value="Queued" action="›" />
      </div>
      <div className="flex flex-wrap gap-2">
        <Chip>filter: active</Chip>
        <Chip variant="outline" onRemove={() => {}}>
          removable
        </Chip>
      </div>
      <div className="flex flex-wrap items-center gap-4 border-t border-border pt-4">
        <StatusDot tone="success" label="API" showLabel />
        <StatusDot tone="warning" pulse label="Throttled" showLabel />
      </div>
      <FoldSection heading="Optional: extra hyperparameters">
        <p className="text-xs">Learning-rate schedule and weight decay overrides.</p>
      </FoldSection>
      <Timeline
        items={[
          { id: 'x1', title: 'Resolve dataset', meta: 'ok' },
          { id: 'x2', title: 'Train', meta: 'in progress' },
        ]}
      />
    </div>
  ),
}
