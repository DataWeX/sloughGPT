import type { Meta, StoryObj } from '@storybook/react'
import { MessageBubble } from './message-bubble'

const meta = {
  title: 'AI/MessageBubble',
  component: MessageBubble,
  tags: ['autodocs'],
  argTypes: {
    role: { control: 'select', options: ['user', 'assistant', 'system'] },
    variant: { control: 'inline-radio', options: ['transcript', 'surface'] },
  },
  decorators: [
    (Story) => (
      <div className="sl-shell-main str-safe-x w-full max-w-lg p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof MessageBubble>

export default meta
type Story = StoryObj<typeof meta>

export const Assistant: Story = {
  args: {
    role: 'assistant',
    children: 'You can lower the learning rate if loss oscillates.',
  },
}

export const User: Story = {
  args: {
    role: 'user',
    children: 'Resume from step_1000.pt and export the adapter.',
  },
}

export const System: Story = {
  args: {
    role: 'system',
    children: 'Tools were reset for this turn.',
  },
}

/** Bordered “card” bubbles — useful in dense dashboards or the component gallery. */
export const SurfaceCards: Story = {
  args: {
    role: 'assistant',
    variant: 'surface',
    children: 'Same content with the older bordered assistant + primary rail.',
  },
}
