import type { Meta, StoryObj } from '@storybook/react'
import { JobStatus, type JobStatusState } from './job-status'

const meta = {
  title: 'AI/JobStatus',
  component: JobStatus,
  tags: ['autodocs'],
} satisfies Meta<typeof JobStatus>

export default meta

const states: JobStatusState[] = ['idle', 'queued', 'running', 'success', 'error', 'cancelled']

export const All: StoryObj = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      {states.map((s) => (
        <JobStatus key={s} status={s} />
      ))}
    </div>
  ),
}
