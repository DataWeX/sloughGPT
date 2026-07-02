import type { Meta, StoryObj } from '@storybook/react'
import { Button } from './button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './card'

const meta = {
  title: 'UI/Card',
  component: Card,
  tags: ['autodocs'],
} satisfies Meta<typeof Card>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <Card className="w-[380px]">
      <CardHeader>
        <CardTitle>Training job</CardTitle>
        <CardDescription>Queued — est. 12m remaining</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">Loss curve is stable; no action required.</p>
      </CardContent>
      <CardFooter className="justify-end gap-2">
        <Button variant="secondary" size="sm">
          Logs
        </Button>
        <Button size="sm">Open</Button>
      </CardFooter>
    </Card>
  ),
}
