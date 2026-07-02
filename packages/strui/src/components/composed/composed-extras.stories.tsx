import type { Meta, StoryObj } from '@storybook/react'
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
  title: 'Composed/Extras',
  tags: ['autodocs'],
} satisfies Meta

export default meta

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
