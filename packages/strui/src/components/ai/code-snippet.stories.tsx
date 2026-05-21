import type { Meta, StoryObj } from '@storybook/react'
import { CodeSnippet } from './code-snippet'

const meta = {
  title: 'AI/CodeSnippet',
  component: CodeSnippet,
  tags: ['autodocs'],
} satisfies Meta<typeof CodeSnippet>

export default meta
type Story = StoryObj<typeof meta>

export const Python: Story = {
  render: () => (
    <CodeSnippet>{`def train_step(batch):
    loss = model(batch)
    loss.backward()
    optim.step()
    return loss.item()`}</CodeSnippet>
  ),
}
