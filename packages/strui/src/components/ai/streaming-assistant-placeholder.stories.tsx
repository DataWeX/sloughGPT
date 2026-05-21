import type { Meta, StoryObj } from '@storybook/react'
import { StreamingAssistantPlaceholder } from './streaming-assistant-placeholder'

const meta = {
  title: 'AI/StreamingAssistantPlaceholder',
  component: StreamingAssistantPlaceholder,
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div className="sl-shell-main p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof StreamingAssistantPlaceholder>

export default meta

export const Default: StoryObj<typeof meta> = {
  args: { lines: 3 },
}
