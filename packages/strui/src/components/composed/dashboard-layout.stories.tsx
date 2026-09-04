import type { Meta, StoryObj } from '@storybook/react'
import { IconCpu, IconDownload, IconRefresh, IconSettings, IconAlert } from '../ui/icons'
import { Button } from '../ui/button'
import { Card, CardContent } from '../ui/card'
import { ActionCard } from './action-card'
import { ChipGroup } from './chip-group'
import { DetailRow } from './detail-row'
import { InfoCard } from './info-card'
import { InsightsCard } from './insights-card'
import { MetricsCard } from './metrics-card'
import { SortDropdown } from './sort-dropdown'
import { TabGroup } from './tab-group'
import { StatCard } from './stat-card'
import { KpiGrid } from './kpi-grid'

const meta = {
  title: 'Composed/DashboardLayout',
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
} satisfies Meta

export default meta

export const FullDashboard: StoryObj = {
  render: () => (
    <div className="max-w-4xl space-y-6 p-4">
      {/* Page Header */}
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

      {/* KPI Row */}
      <MetricsCard title="Overview" columns={4}>
        <StatCard label="Total Jobs" value={42} />
        <StatCard label="Active" value={3} trend={{ value: 12, positive: true }} />
        <StatCard label="Failed" value={1} trend={{ value: 5, positive: false }} />
        <StatCard label="Avg Time" value="2.4h" />
      </MetricsCard>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Health */}
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

        {/* Feedback Insights */}
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

      {/* Details Section */}
      <Card>
        <CardContent className="p-4">
          <h3 className="text-sm font-medium mb-3">Job Details</h3>
          <div className="space-y-2">
            <DetailRow label="Job ID" value="job_01a2b3c4d" mono />
            <DetailRow label="Model" value="GPT-2 (124M)" />
            <DetailRow label="Dataset" value="sft-v1.jsonl" />
            <DetailRow label="Status" value="Running" valueClassName="text-green-500" />
            <DetailRow label="Started" value="2 hours ago" />
          </div>
        </CardContent>
      </Card>

      {/* Tags */}
      <Card>
        <CardContent className="p-4">
          <h3 className="text-sm font-medium mb-3">Tags</h3>
          <ChipGroup
            chips={[
              { label: 'SFT', tone: 'primary' },
              { label: 'GPT-2', tone: 'success' },
              { label: 'GPU', tone: 'warning' },
              { label: 'Production', tone: 'default' },
            ]}
          />
        </CardContent>
      </Card>

      {/* Tabs */}
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
