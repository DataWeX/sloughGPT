import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { PromptComposer } from './prompt-composer'

const meta = {
  title: 'AI/PromptComposer',
  component: PromptComposer,
  tags: ['autodocs'],
  argTypes: {
    busy: { control: 'boolean' },
    disabled: { control: 'boolean' },
    safeAreaBottom: { control: 'boolean' },
    sendLabel: { control: 'text' },
    placeholder: { control: 'text' },
  },
  parameters: {
    docs: {
      description: {
        component:
          'Bottom prompt form with safe-area padding (`str-safe-bottom`), touch-sized send, and optional `textareaRef` / `textAreaProps` / `sendButtonProps` for host apps (e.g. Next.js chat).',
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="sl-shell-main flex min-h-[280px] w-full max-w-lg flex-col justify-end rounded-none border border-border">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof PromptComposer>

export default meta

function Stateful() {
  const [value, setValue] = useState('')
  return (
    <PromptComposer
      value={value}
      onValueChange={setValue}
      onSubmit={() => setValue('')}
      placeholder="Ask anything…"
    />
  )
}

export const Interactive: StoryObj = {
  render: () => <Stateful />,
}

export const Busy: StoryObj = {
  args: {
    busy: true,
  },
  render: () => (
    <PromptComposer
      value="Streaming in progress…"
      onValueChange={() => {}}
      onSubmit={() => {}}
      busy
      placeholder="Message…"
    />
  ),
}

function NoSafeAreaStory() {
  const [value, setValue] = useState('')
  return (
    <PromptComposer
      safeAreaBottom={false}
      value={value}
      onValueChange={setValue}
      onSubmit={() => setValue('')}
      placeholder="Use when the shell already applies env(safe-area-inset-bottom)."
    />
  )
}

export const WithoutSafeAreaPadding: StoryObj = {
  name: 'No safe-area (host owns inset)',
  render: () => <NoSafeAreaStory />,
}
