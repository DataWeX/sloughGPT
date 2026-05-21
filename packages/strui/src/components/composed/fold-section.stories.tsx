import type { Meta, StoryObj } from '@storybook/react'

import { FoldSection } from './fold-section'

const meta = {
  title: 'Composed/FoldSection',
  component: FoldSection,
  tags: ['autodocs'],
} satisfies Meta<typeof FoldSection>

export default meta

type Story = StoryObj<typeof FoldSection>

export const Default: Story = {
  args: {
    heading: 'Advanced tokenizer settings',
    children: (
      <p className="text-muted-foreground">
        Options for BPE merges, normalization, and special tokens appear here when the backend exposes them.
      </p>
    ),
    open: true,
  },
}

export const Stack: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-lg space-y-2 p-4">
      <FoldSection heading="Dataset manifest">
        <p>Point to a JSON manifest or use a registered dataset ref.</p>
      </FoldSection>
      <FoldSection heading="Environment overrides">
        <p className="font-mono text-xs">CUDA_VISIBLE_DEVICES=0</p>
      </FoldSection>
    </div>
  ),
}
