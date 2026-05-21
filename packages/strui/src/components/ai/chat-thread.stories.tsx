import type { Meta, StoryObj } from '@storybook/react'
import { ChatThread } from './chat-thread'
import { MessageBubble } from './message-bubble'
import { TypingIndicator } from './typing-indicator'

const meta = {
  title: 'AI/ChatThread',
  component: ChatThread,
  tags: ['autodocs'],
  argTypes: {
    density: {
      control: 'inline-radio',
      options: ['comfortable', 'compact'],
    },
  },
  decorators: [
    (Story) => (
      <div className="sl-shell-main flex h-[420px] w-full max-w-lg flex-col overflow-hidden rounded-none border border-border">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ChatThread>

export default meta
type Story = StoryObj<typeof meta>

export const WithMessages: Story = {
  render: () => (
    <ChatThread className="min-h-0 flex-1">
      <MessageBubble role="user">Summarize the last eval run.</MessageBubble>
      <MessageBubble role="assistant">
        Perplexity improved by 8% vs baseline; no regressions on the held-out set.
      </MessageBubble>
      <TypingIndicator />
    </ChatThread>
  ),
}
