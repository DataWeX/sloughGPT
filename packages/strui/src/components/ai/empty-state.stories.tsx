import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '../ui/button'
import { EmptyState } from './empty-state'

const meta = {
  title: 'AI/EmptyState',
  component: EmptyState,
  tags: ['autodocs'],
} satisfies Meta<typeof EmptyState>

export default meta
type Story = StoryObj<typeof meta>

export const Chat: Story = {
  args: {
    title: 'Start a conversation',
    description: 'Ask about training jobs, models, or paste a config to review.',
  },
  render: (args) => (
    <EmptyState {...args}>
      <Button className="str-touch-target w-full">New chat</Button>
    </EmptyState>
  ),
}
