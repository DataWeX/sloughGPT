import type { Meta, StoryObj } from '@storybook/react'
import { TokenMeter } from './token-meter'

const meta = {
  title: 'AI/TokenMeter',
  component: TokenMeter,
  tags: ['autodocs'],
} satisfies Meta<typeof TokenMeter>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    total: 12400,
    contextLimit: 128000,
  },
}

export const SmallContext: Story = {
  args: {
    total: 3800,
    contextLimit: 4096,
  },
}
