'use client'

import { useRef, useState } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Progress,
  cn,
} from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'
import { formatToastError } from '@/lib/error-utils'
import type { InputMode } from '@/hooks/useTrainingForm'

/**
 * "How do you want to train?" — inline mode chooser for the /training
 * journey page (card 20261119_130319).
 *
 * The page only surfaced datasets, so the other ways in were invisible.
 * This widget names them. Two rules it exists to hold:
 *
 *  - **No dead buttons.** Every card either hands off to a flow that
 *    already exists on this page (guided setup, fast-train fold) or runs
 *    training through an entry the API already serves. Ways without a
 *    web-reachable entry (DPO, chat-train, curriculum) stay off the
 *    chooser rather than becoming a button that does nothing.
 *  - **One path to Start.** The dataset and paste-text cards only
 *    *configure* the guided setup below — they never launch a run of
 *    their own. There is exactly one Start for the form.
 *
 * Copy is plain language (UX_FLOWS Feature #12): methods are folded into
 * the descriptions instead of shown as jargon, and the native-vs-external
 * model axis is named where it helps someone choose.
 */

export type TrainingModeId = 'dataset' | 'text' | 'quick' | 'chats'

export interface TrainingModeChooserProps {
  /** Current guided-setup data source — drives the "you are here" cue. */
  inputMode: InputMode
  /** Dataset card → set data source to files and open the Data step. */
  onPickDataset: () => void
  /** Paste-text card → set data source to pasted text and open Configure. */
  onPickText: () => void
  /** Quick card → open the existing "Fast train alternative" fold. */
  onPickQuick: () => void
}

type ChatsPhase = 'idle' | 'starting' | 'running' | 'complete' | 'error'

interface ChatsRun {
  phase: ChatsPhase
  progress: number | null
  message: string | null
  cancelled: boolean
}

const IDLE: ChatsRun = { phase: 'idle', progress: null, message: null, cancelled: false }

function chatsLabel(run: ChatsRun): string {
  switch (run.phase) {
    case 'starting':
      return 'Getting your conversations ready…'
    case 'running':
      return 'Learning from your conversations…'
    case 'complete':
      return run.cancelled
        ? 'Stopped — no new saved version'
        : 'Finished — your new saved version is ready'
    case 'error':
      return 'Could not train from your chats'
    default:
      return ''
  }
}

export function TrainingModeChooser({
  inputMode,
  onPickDataset,
  onPickText,
  onPickQuick,
}: TrainingModeChooserProps) {
  const [chats, setChats] = useState<ChatsRun>(IDLE)
  const cancelledRef = useRef(false)

  async function startChats() {
    if (chats.phase === 'starting' || chats.phase === 'running') return
    cancelledRef.current = false
    setChats({ ...IDLE, phase: 'starting' })
    try {
      await trainingJobsController.startFromSessionsSloNet({})
      if (cancelledRef.current) return
      setChats({ ...IDLE, phase: 'running' })

      let sawTerminal = false
      for await (const event of trainingJobsController.streamFromSessionsSloNet()) {
        if (cancelledRef.current) break
        const raw = event.data?.progress_percent
        const progress = typeof raw === 'number' ? raw : null
        const done = event.status === 'complete' || event.status === 'error'
        if (done) sawTerminal = true
        setChats({
          phase: event.status === 'complete' ? 'complete' : event.status === 'error' ? 'error' : 'running',
          progress,
          message: event.message || null,
          cancelled: event.data?.cancelled === true,
        })
        if (done) break
      }
      // Stream closed without a terminal frame — the run is over either way;
      // never leave the card stuck on "Learning…".
      if (!sawTerminal && !cancelledRef.current) {
        setChats((run) => (run.phase === 'running' ? { ...run, phase: 'complete' } : run))
      }
    } catch (err) {
      if (cancelledRef.current) return
      setChats({
        ...IDLE,
        phase: 'error',
        message: formatToastError(err, 'Could not train from your chats'),
      })
    }
  }

  async function stopChats() {
    cancelledRef.current = true
    try {
      await trainingJobsController.cancelFromSessionsSloNet()
    } catch {
      // Best effort — the run is dropped locally either way.
    }
    setChats(IDLE)
  }

  const activeMode: TrainingModeId =
    chats.phase === 'starting' || chats.phase === 'running'
      ? 'chats'
      : inputMode === 'text'
        ? 'text'
        : 'dataset'

  const modes: {
    id: TrainingModeId
    title: string
    description: string
    pick: () => void
  }[] = [
    {
      id: 'dataset',
      title: 'Train on a dataset',
      description:
        'Pick files you already have — text or images. Train from scratch, continue an outside model, or build a native SloNet.',
      pick: onPickDataset,
    },
    {
      id: 'text',
      title: 'Paste text',
      description: 'No files yet? Paste stories, docs, conversations or code straight into the box.',
      pick: onPickText,
    },
    {
      id: 'quick',
      title: 'Quick train',
      description: 'A short run with sensible defaults — the fastest way to see it work.',
      pick: onPickQuick,
    },
    {
      id: 'chats',
      title: 'Train from your chats',
      description: 'Learn from your conversation history — no files needed.',
      pick: startChats,
    },
  ]

  const showRun = chats.phase !== 'idle'

  return (
    <Card data-testid="training-mode-chooser">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">How do you want to train?</CardTitle>
        <CardDescription>
          Four ways in — pick one and the guided setup below follows.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          {modes.map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={mode.pick}
              aria-current={activeMode === mode.id ? 'true' : undefined}
              className={cn(
                'flex flex-col items-start gap-1 rounded-lg border p-4 text-left',
                'transition-colors hover:border-primary/40 hover:bg-card/70',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                activeMode === mode.id
                  ? 'border-primary/40 bg-primary/5'
                  : 'border-border/60 bg-card/40',
              )}
            >
              <span className="text-sm font-medium text-foreground">{mode.title}</span>
              <span className="text-xs text-muted-foreground">{mode.description}</span>
            </button>
          ))}
        </div>

        {showRun && (
          <div
            role="status"
            aria-live="polite"
            data-testid="from-chats-status"
            className="space-y-2 rounded-lg border border-border/60 bg-card/40 p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-foreground">{chatsLabel(chats)}</span>
              {(chats.phase === 'starting' || chats.phase === 'running') && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-6 px-2 text-xs"
                  onClick={() => void stopChats()}
                >
                  Stop
                </Button>
              )}
            </div>
            {(chats.phase === 'starting' || chats.phase === 'running') && (
              <Progress
                value={chats.progress}
                indeterminate={chats.phase === 'starting'}
                label="Chat training progress"
                showValue
                size="sm"
              />
            )}
            {chats.phase === 'running' && chats.message && (
              <p className="text-xs text-muted-foreground">{chats.message}</p>
            )}
            {chats.phase === 'error' && chats.message && (
              <p className="text-xs text-destructive">{chats.message}</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
