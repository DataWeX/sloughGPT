import type { Meta, StoryObj } from '@storybook/react'

import { Breadcrumbs } from './breadcrumbs'

const meta = {
  title: 'Composed/Breadcrumbs',
  component: Breadcrumbs,
  tags: ['autodocs'],
} satisfies Meta<typeof Breadcrumbs>

export default meta

type Story = StoryObj<typeof Breadcrumbs>

export const Default: Story = {
  args: {
    items: [
      { label: 'Workspace', href: '#' },
      { label: 'Datasets', href: '#' },
      { label: 'shakespeare' },
    ],
  },
}

export const PathLike: StoryObj = {
  render: () => (
    <div className="str-safe-x p-4 font-mono text-xs">
      <Breadcrumbs
        separator="›"
        items={[
          { label: 'datasets', href: '#' },
          { label: 'runs', href: '#' },
          { label: '2026-04-05-143022' },
        ]}
      />
    </div>
  ),
}
