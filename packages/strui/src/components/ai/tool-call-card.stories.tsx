import type { Meta, StoryObj } from '@storybook/react'
import { ToolCallCard } from './tool-call-card'

const meta = {
  title: 'AI/ToolCallCard',
  component: ToolCallCard,
  tags: ['autodocs'],
  argTypes: {
    state: { control: 'select', options: ['pending', 'ok', 'error'] },
  },
} satisfies Meta<typeof ToolCallCard>

export default meta
type Story = StoryObj<typeof meta>

export const Ok: Story = {
  args: {
    name: 'search_docs',
    argsPreview: '{"query": "checkpoint resume", "top_k": 3}',
    state: 'ok',
  },
}

export const Pending: Story = {
  args: {
    name: 'run_sql',
    argsPreview: '{"sql": "SELECT * FROM jobs LIMIT 5"}',
    state: 'pending',
  },
}

export const Error: Story = {
  args: {
    name: 'fetch_url',
    argsPreview: '{"url": "https://example.com"}',
    state: 'error',
  },
}
