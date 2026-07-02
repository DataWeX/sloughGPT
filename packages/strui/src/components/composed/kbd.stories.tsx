import type { Meta, StoryObj } from '@storybook/react'

import { Kbd } from './kbd'

const meta = {
  title: 'Composed/Kbd',
  component: Kbd,
  tags: ['autodocs'],
} satisfies Meta<typeof Kbd>

export default meta

type Story = StoryObj<typeof Kbd>

export const Default: Story = {
  args: { children: '⌘K' },
}

export const Combo: StoryObj = {
  render: () => (
    <p className="str-safe-x flex flex-wrap items-center gap-1 p-4 text-sm text-muted-foreground">
      Press <Kbd>Ctrl</Kbd>
      <span>+</span>
      <Kbd>Enter</Kbd>
      <span>to send</span>
    </p>
  ),
}
