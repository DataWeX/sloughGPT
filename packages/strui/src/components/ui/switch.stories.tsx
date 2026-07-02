import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { Label } from './label'
import { Switch } from './switch'

const meta = {
  title: 'UI/Switch',
  component: Switch,
  tags: ['autodocs'],
} satisfies Meta<typeof Switch>

export default meta

export const Default: StoryObj<typeof Switch> = {
  args: {
    defaultChecked: false,
  },
}

export const WithLabel: StoryObj = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <div className="flex items-center gap-3">
        <Switch id="airplane" checked={on} onCheckedChange={setOn} aria-labelledby="airplane-label" />
        <Label id="airplane-label" htmlFor="airplane">
          Stream tokens
        </Label>
      </div>
    )
  },
}
