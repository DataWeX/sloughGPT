'use client'

import { forwardRef, useEffect, useMemo } from 'react'
import { TrainingMessageBubble, TrainingTurn, type TrainingMessage } from './TrainingMessageBubble'

interface TrainingMessagesProps {
  messages: TrainingMessage[]
  className?: string
}

export const TrainingMessages = forwardRef<HTMLDivElement, TrainingMessagesProps>(
  function TrainingMessages({ messages, className }, ref) {
    // Group messages by turn (3 messages per turn)
    const turns = useMemo(() => {
      const grouped: TrainingMessage[][] = []
      for (let i = 0; i < messages.length; i += 3) {
        grouped.push(messages.slice(i, i + 3))
      }
      return grouped
    }, [messages])
    
return (
      <section className="flex-1 min-h-0 overflow-y-auto">
        <div className="mx-auto max-w-2xl px-3 py-4 sm:px-4 sm:py-6 space-y-2">
          {messages.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <div className="text-muted-foreground">
                Ready to train!
              </div>
              <div className="text-xs text-muted-foreground">
                Teacher prompts -&gt; Baby responds -&gt; Teacher corrects -&gt; Baby learns
              </div>
            </div>
          ) : (
            turns.map((turnMessages, idx) => (
              <TrainingTurn key={idx} messages={turnMessages} />
            ))
          )}
          <div ref={ref} />
        </div>
      </section>
    )
  }
)