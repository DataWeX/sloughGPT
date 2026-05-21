import type { Meta, StoryObj } from '@storybook/react'

import { Timeline } from './timeline'

const meta = {
  title: 'Composed/Timeline',
  component: Timeline,
  tags: ['autodocs'],
} satisfies Meta<typeof Timeline>

export default meta

type Story = StoryObj<typeof Timeline>

export const TrainingPhases: Story = {
  args: {
    items: [
      { id: '1', title: 'Corpus resolved', meta: 'manifest v3 · 12:01' },
      { id: '2', title: 'Warmup steps', meta: 'steps 0–500' },
      { id: '3', title: 'Training', meta: 'epoch 2 / 4 · running' },
      { id: '4', title: 'Export checkpoint', meta: 'pending' },
    ],
  },
}

export const InPanel: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-md rounded-none border border-border bg-card/40 p-4">
      <p className="mb-4 text-xs font-medium uppercase tracking-wider text-muted-foreground">Run timeline</p>
      <Timeline
        items={[
          { id: 'a', title: 'Queued', meta: 'job_7f3…', tone: 'muted' },
          { id: 'b', title: 'Running', meta: 'loss 2.41', tone: 'warning' },
          { id: 'c', title: 'Completed', meta: 'checkpoint written', tone: 'success' },
        ]}
      />
    </div>
  ),
}
