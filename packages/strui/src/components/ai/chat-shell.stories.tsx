import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '../ui/button'
import { ChatLayout } from './chat-layout'
import { ChatThread } from './chat-thread'
import { CodeSnippet } from './code-snippet'
import { EmptyState } from './empty-state'
import { MessageBubble } from './message-bubble'
import { PromptComposer } from './prompt-composer'
import { TokenMeter } from './token-meter'
import { ToolCallCard } from './tool-call-card'
import { TypingIndicator } from './typing-indicator'

const meta = {
  title: 'AI/ChatShell',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story:
          'Full-height chat shell using **ChatLayout**: header, **ChatThread**, **PromptComposer**. Rich blocks (tool + code) appear only after the assistant reply. **iPhone** uses the Storybook viewport preset.',
      },
    },
  },
} satisfies Meta

export default meta

type DemoMsg =
  | { role: 'user'; text: string }
  | { role: 'assistant'; text: string; showRich?: boolean }

function ShellDemo() {
  const [value, setValue] = useState('')
  const [messages, setMessages] = useState<DemoMsg[]>([])
  const [busy, setBusy] = useState(false)

  const send = () => {
    const t = value.trim()
    if (!t) return
    setMessages((m) => [...m, { role: 'user', text: t }])
    setValue('')
    setBusy(true)
    window.setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: 'Acknowledged. Here is a compact checkpoint read and a loss line from the run.',
          showRich: true,
        },
      ])
      setBusy(false)
    }, 650)
  }

  const header = (
    <header className="str-safe-top str-safe-x flex flex-wrap items-start justify-between gap-3 px-3 py-3.5 sm:px-5">
      <div className="min-w-0 space-y-0.5">
        <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-primary">Session</p>
        <h1 className="text-base font-semibold leading-tight text-foreground">Assistant</h1>
        <p className="text-xs text-muted-foreground">Pastel lattice · sharp geometry</p>
      </div>
      <TokenMeter total={8420} contextLimit={128000} />
    </header>
  )

  const thread =
    messages.length === 0 ? (
      <EmptyState
        className="min-h-0 flex-1 justify-center"
        title="Start a conversation"
        description="Messages scroll in the thread above a fixed composer. Safe areas apply on notched devices."
      >
        <Button type="button" variant="secondary" className="str-touch-target w-full" onClick={() => {}}>
          Browse models
        </Button>
      </EmptyState>
    ) : (
      <ChatThread className="min-h-0 flex-1">
        {messages.map((msg, i) =>
          msg.role === 'user' ? (
            <MessageBubble key={i} role="user">
              {msg.text}
            </MessageBubble>
          ) : (
            <div key={i} className="flex max-w-[var(--chat-thread-max)] flex-col gap-3">
              <MessageBubble role="assistant">{msg.text}</MessageBubble>
              {msg.showRich ? (
                <>
                  <ToolCallCard
                    name="list_checkpoints"
                    argsPreview='{"dir": "data/experiments/run_01"}'
                    state="ok"
                  />
                  <CodeSnippet>{'loss = criterion(logits, labels)'}</CodeSnippet>
                </>
              ) : null}
            </div>
          ),
        )}
        {busy ? <TypingIndicator /> : null}
      </ChatThread>
    )

  const composer = (
    <PromptComposer
      value={value}
      onValueChange={setValue}
      onSubmit={send}
      busy={busy}
      placeholder="Message…"
    />
  )

  return <ChatLayout header={header} thread={thread} composer={composer} />
}

export const Desktop: StoryObj = {
  render: () => <ShellDemo />,
}

export const IPhone: StoryObj = {
  ...Desktop,
  parameters: {
    viewport: {
      defaultViewport: 'iphone12',
    },
  },
}
