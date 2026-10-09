import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { IconCpu, IconDownload, IconRefresh, IconSettings, IconAlert } from '../ui/icons'
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
import { DetailRow } from './detail-row'
import { InfoCard } from './info-card'
import { InsightsCard } from './insights-card'
import { MetricsCard } from './metrics-card'
import { SortDropdown } from './sort-dropdown'
import { TabGroup } from './tab-group'
import { StatusBadge } from './status-badge'
import { DetailList } from './detail-list'
import { LoadingCard } from './loading-card'
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
          <NavRail header={<span className="px-3 text-xs font-bold uppercase tracking-wider">Acme</span>}>
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
            <StatCard label="Requests" value="12.4k" icon="📈" trend={{ value: 3, positive: true }} />
            <StatCard label="Latency p95" value="420ms" />
            <StatCard label="Errors" value="0.02%" />
            <StatCard label="Cost" value="$48" />
          </KpiGrid>
          <div className="grid gap-6 lg:grid-cols-2">
            <EmptyCard message="No experiments" action={<Button className="str-touch-target w-full sm:w-auto">Create experiment</Button>} />
            <div className="space-y-3">
              <p className="text-xs font-medium uppercase text-muted-foreground">Loading skeleton</p>
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
          <SortDropdown
            value="newest"
            options={[
              { value: 'newest', label: 'Newest' },
              { value: 'oldest', label: 'Oldest' },
              { value: 'status', label: 'Status' },
            ]}
            onChange={() => {}}
          />
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
        <ActionCard
          title="System Health"
          subtitle="Real-time metrics"
          actions={
            <Button size="sm" variant="ghost">
              <IconRefresh className="h-3 w-3" />
            </Button>
          }
        >
          <KpiGrid columns={2}>
            <InfoCard
              icon={<IconCpu className="h-4 w-4" />}
              title="GPU"
              description="NVIDIA RTX 4090"
              tone="success"
              size="sm"
            />
            <InfoCard
              icon={<IconAlert className="h-4 w-4" />}
              title="Memory"
              description="16 / 32 GB"
              tone="warning"
              size="sm"
            />
          </KpiGrid>
        </ActionCard>

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

      <TabGroup
        defaultValue="logs"
        tabs={[
          {
            value: 'logs',
            label: 'Training Logs',
            content: (
              <div className="font-mono text-xs space-y-1 py-2">
                <div className="text-muted-foreground">[00:12:34] Epoch 1/10 — loss: 0.45</div>
                <div className="text-muted-foreground">[00:12:45] Epoch 2/10 — loss: 0.38</div>
                <div className="text-muted-foreground">[00:12:56] Epoch 3/10 — loss: 0.31</div>
                <div className="text-green-500">[00:13:07] Epoch 4/10 — loss: 0.23 ✓</div>
              </div>
            ),
          },
          {
            value: 'config',
            label: 'Configuration',
            content: (
              <div className="py-2 space-y-2">
                <DetailRow label="Learning Rate" value="2e-4" />
                <DetailRow label="Batch Size" value="32" />
                <DetailRow label="Epochs" value="10" />
              </div>
            ),
          },
          {
            value: 'metrics',
            label: 'Metrics',
            content: <p className="text-sm py-2">Metrics visualization coming soon.</p>,
          },
        ]}
      />
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

// ── InfoCard ───────────────────────────────────────────────────────

export const InfoCardDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4 space-y-3">
      <InfoCard
        icon={<IconCpu className="h-5 w-5" />}
        title="GPU Acceleration"
        description="CUDA cores available"
        tone="success"
      >
        <p className="text-xs text-muted-foreground">Device: NVIDIA RTX 4090</p>
      </InfoCard>
      <InfoCard
        icon={<IconDownload className="h-5 w-5" />}
        title="Model Download"
        description="GPT-2 (500 MB)"
        tone="primary"
      />
      <InfoCard title="No description" tone="muted" />
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
        chips={[
          { label: 'Filter: active' },
          { label: 'Status: running', tone: 'success' },
        ]}
        gap={2}
      />
    </div>
  ),
}

// ── DetailRow ──────────────────────────────────────────────────────

export const DetailRowDemo: StoryObj = {
  render: () => (
    <div className="max-w-sm p-4 space-y-3">
      <DetailRow label="Model" value="GPT-2" />
      <DetailRow label="Path" value="/models/gpt2" mono />
      <DetailRow label="Size" value="500 MB" valueClassName="text-muted-foreground" />
      <DetailRow label="Docs" value="View" href="https://example.com" />
    </div>
  ),
}

// ── SortDropdown ───────────────────────────────────────────────────

export const SortDropdownDemo: StoryObj = {
  render: () => (
    <div className="p-4">
      <SortDropdown
        value="newest"
        options={[
          { value: 'newest', label: 'Newest' },
          { value: 'oldest', label: 'Oldest' },
          { value: 'importance', label: 'Importance' },
        ]}
        onChange={() => {}}
      />
    </div>
  ),
}

// ── TabGroup ───────────────────────────────────────────────────────

export const TabGroupDemo: StoryObj = {
  render: () => (
    <div className="max-w-lg p-4">
      <TabGroup
        defaultValue="overview"
        tabs={[
          { value: 'overview', label: 'Overview', content: <p className="text-sm">Overview content here.</p> },
          { value: 'details', label: 'Details', content: <p className="text-sm">Details content here.</p> },
          { value: 'logs', label: 'Logs', content: <p className="text-sm">Logs content here.</p> },
        ]}
      />
    </div>
  ),
}

export const TabGroupPills: StoryObj = {
  render: () => (
    <div className="max-w-lg p-4">
      <TabGroup
        defaultValue="all"
        layout="pills"
        tabs={[
          { value: 'all', label: 'All', content: <p className="text-sm">All items.</p> },
          { value: 'active', label: 'Active', content: <p className="text-sm">Active items.</p> },
          { value: 'archived', label: 'Archived', content: <p className="text-sm">Archived items.</p> },
        ]}
      />
    </div>
  ),
}

export const TabGroupBoxed: StoryObj = {
  render: () => (
    <div className="max-w-lg p-4">
      <TabGroup
        defaultValue="train"
        layout="boxed"
        tabs={[
          { value: 'train', label: 'Train', content: <p className="text-sm">Training config.</p> },
          { value: 'eval', label: 'Eval', content: <p className="text-sm">Evaluation config.</p> },
          { value: 'export', label: 'Export', content: <p className="text-sm">Export config.</p> },
        ]}
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

// ── DetailList ─────────────────────────────────────────────────────

export const DetailListDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4">
      <DetailList
        items={[
          { label: 'Model', value: 'GPT-2' },
          { label: 'Path', value: '/models/gpt2', mono: true },
          { label: 'Size', value: '500 MB' },
          { label: 'Docs', value: 'View', href: 'https://example.com' },
        ]}
      />
    </div>
  ),
}

// ── LoadingCard ────────────────────────────────────────────────────

export const LoadingCardDemo: StoryObj = {
  render: () => (
    <div className="max-w-md p-4">
      <LoadingCard title="Loading data..." rows={3} />
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
