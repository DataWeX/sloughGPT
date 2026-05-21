import type { Meta, StoryObj } from '@storybook/react'

import { StatusDot } from './status-dot'

const meta = {
  title: 'Composed/StatusDot',
  component: StatusDot,
  tags: ['autodocs'],
  argTypes: {
    tone: { control: 'select', options: ['success', 'warning', 'destructive', 'muted', 'primary'] },
  },
} satisfies Meta<typeof StatusDot>

export default meta

type Story = StoryObj<typeof StatusDot>

export const Default: Story = {
  args: { tone: 'success', label: 'API reachable', showLabel: true },
}

export const Pulsing: Story = {
  args: { tone: 'primary', pulse: true, label: 'Streaming', showLabel: true },
}

export const Row: StoryObj = {
  render: () => (
    <div className="str-safe-x flex flex-wrap gap-6 p-4 text-sm">
      <StatusDot tone="success" label="Healthy" showLabel />
      <StatusDot tone="warning" label="Degraded" showLabel />
      <StatusDot tone="destructive" label="Down" showLabel />
      <StatusDot tone="muted" label="Unknown" showLabel />
      <StatusDot tone="primary" pulse label="Live" showLabel />
    </div>
  ),
}
