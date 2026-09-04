import type { Meta, StoryObj } from '@storybook/react'
import { IconCpu, IconDownload, IconRefresh, IconSettings } from '../ui/icons'
import { Button } from '../ui/button'
import { ActionCard } from './action-card'
import { CardDialog } from './card-dialog'
import { ChipGroup } from './chip-group'
import { DetailRow } from './detail-row'
import { InfoCard } from './info-card'
import { InsightsCard } from './insights-card'
import { MetricsCard } from './metrics-card'
import { SortDropdown } from './sort-dropdown'
import { TabGroup } from './tab-group'

const meta = {
  title: 'Composed/NewComposites',
  tags: ['autodocs'],
} satisfies Meta

export default meta

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

// ── ConfirmDialog ──────────────────────────────────────────────────
// ConfirmDialog requires a portal, shown in composed-overview.stories.tsx

// ── ErrorBoundary ──────────────────────────────────────────────────
// ErrorBoundary requires runtime error, shown in composed-overview.stories.tsx

// ── CardDialog ─────────────────────────────────────────────────────
// CardDialog requires a portal, shown in composed-overview.stories.tsx

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
      <InfoCard
        title="No description"
        tone="muted"
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
