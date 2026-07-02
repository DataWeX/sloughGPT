import type { Meta, StoryObj } from '@storybook/react'
import { Button } from './button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './dropdown-menu'

const meta = {
  title: 'UI/DropdownMenu',
  tags: ['autodocs'],
} satisfies Meta

export default meta

export const Default: StoryObj = {
  render: () => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="str-touch-target">
          Model
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="str-safe-x">
        <DropdownMenuItem className="str-touch-target">gpt2</DropdownMenuItem>
        <DropdownMenuItem className="str-touch-target">Llama 3</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem className="str-touch-target" disabled>
          Unavailable
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  ),
}
