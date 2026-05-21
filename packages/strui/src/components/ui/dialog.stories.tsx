import type { Meta, StoryObj } from '@storybook/react'
import { Button } from './button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from './dialog'

const meta = {
  title: 'UI/Dialog',
  tags: ['autodocs'],
} satisfies Meta

export default meta

export const Default: StoryObj = {
  render: () => (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="secondary">Open dialog</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Confirm action</DialogTitle>
          <DialogDescription>This matches the apps/web dialog shell — safe on narrow viewports.</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="ghost" className="str-touch-target">
            Cancel
          </Button>
          <Button className="str-touch-target">Continue</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  ),
}
