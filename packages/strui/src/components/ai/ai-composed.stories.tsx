import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { MessageBubble } from './message-bubble'
import { AttachmentChip } from './attachment-chip'
import { ChatLayout } from './chat-layout'
import { ChatThread } from './chat-thread'
import { Citation } from './citation'
import { ModelPicker } from './model-picker'
import { PromptComposer } from './prompt-composer'
import { ReasoningPanel } from './reasoning-panel'
import { SourceList } from './source-list'
import { TokenMeter } from './token-meter'
import { ToolCallCard } from './tool-call-card'
import { TypingIndicator } from './typing-indicator'

const meta = {
  title: 'AI/Composed flows',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story:
          'End-to-end agent chat: model picker, token meter, citations, reasoning, tools, sources, attachments, and composer. Use the Surface toolbar to check light and dark contrast.',
      },
    },
  },
} satisfies Meta

export default meta

const models = [
  { id: 'gpt2', label: 'GPT-2', badge: 'local' },
  { id: 'llama', label: 'Llama 3.1 8B', badge: 'fast' },
]

export const AgentChat: StoryObj = {
  render: () => {
    const [model, setModel] = useState('gpt2')
    const [prompt, setPrompt] = useState('')
    return (
      <ChatLayout
        header={
          <div className="str-safe-top str-safe-x flex flex-wrap items-center justify-between gap-3 border-b border-border bg-card/80 px-3 py-3">
            <ModelPicker value={model} options={models} onChange={setModel} fullWidth={false} />
            <TokenMeter total={4200} contextLimit={8192} />
          </div>
        }
        thread={
          <ChatThread className="min-h-0 flex-1">
            <MessageBubble role="user">
              <span>
                Summarize with citations{' '}
                <Citation index={1} href="https://example.com/doc" />
              </span>
            </MessageBubble>
            <ReasoningPanel title="Reasoning (collapsed)">Planning retrieval steps…</ReasoningPanel>
            <ToolCallCard name="search" argsPreview='{"q":"docs"}' state="ok" />
            <MessageBubble role="assistant">
              <span>
                Here is the answer referencing the doc{' '}
                <Citation index={1} href="https://example.com/doc" />.
              </span>
            </MessageBubble>
            <SourceList
              sources={[
                { title: 'Product guide', url: 'https://example.com/doc', snippet: '…' },
                { title: 'Internal wiki', url: 'https://example.com/wiki' },
              ]}
            />
            <TypingIndicator />
          </ChatThread>
        }
        composer={
          <div className="space-y-2 bg-background px-3 pt-2">
            <div className="flex flex-wrap gap-2">
              <AttachmentChip name="screenshot.png" onRemove={() => {}} />
            </div>
            <PromptComposer
              value={prompt}
              onValueChange={setPrompt}
              onSubmit={() => setPrompt('')}
              placeholder="Ask with attachments…"
            />
          </div>
        }
      />
    )
  },
}
