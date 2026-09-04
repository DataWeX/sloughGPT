'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Button, cn } from '@sloughgpt/strui'
import { IconInfo, IconX } from '@sloughgpt/strui'

export interface StepHint {
  title: string
  content: string
  tip?: string
}

const STEP_HINTS: Record<string, StepHint> = {
  'data-selector': {
    title: 'Dataset selection',
    content: 'Choose an existing dataset or import a new one. JSONL conversation format trains best — each line should be a {"messages": [...]} object.',
    tip: 'Start with a small dataset (100-500 samples) to test before committing to a full training run.',
  },
  'data-kaggle': {
    title: 'Quick import from Kaggle',
    content: 'One-click import from Kaggle Hub. Requires API credentials configured on the server (~/.kaggle/kaggle.json).',
    tip: 'The Titanic and Iris datasets are great for testing — they import in seconds.',
  },
  'data-huggingface': {
    title: 'Quick import from HuggingFace',
    content: 'Import datasets from HuggingFace Hub. No server setup required — works out of the box.',
    tip: 'Tiny Shakespeare is a classic for testing text generation.',
  },
  'data-paste': {
    title: 'Paste text directly',
    content: 'Paste any text to train on — stories, documentation, conversations, code. The system will chunk it into training samples automatically.',
    tip: 'Paste at least 1000 characters for meaningful training results.',
  },
  'data-preview': {
    title: 'Dataset preview',
    content: 'Preview the first few samples to verify the data format looks correct before training.',
  },
  'configure-method': {
    title: 'Training method',
    content: 'Train from scratch: distills knowledge into a small model. Continue training: fine-tunes an existing model. Native SloNet: trains a pure transformer from scratch.',
    tip: 'Start with "Train from scratch" — it\'s the simplest and works well for most use cases.',
  },
  'configure-hyperparams': {
    title: 'Hyperparameters',
    content: 'Epochs: how many times to loop through the data. Batch size: samples per gradient step. Learning rate: step size for optimization.',
    tip: 'Default values work well for most datasets. Only adjust if you know what you\'re doing.',
  },
  'configure-lora': {
    title: 'LoRA (Low-Rank Adaptation)',
    content: 'Parameter-efficient fine-tuning. Only trains a small number of additional parameters, reducing memory usage by ~60% while maintaining quality.',
    tip: 'Enable LoRA for fine-tuning large models — it\'s faster and uses less memory.',
  },
  'configure-checkpoint': {
    title: 'Resume from checkpoint',
    content: 'Continue training from a previously saved checkpoint instead of starting fresh. Useful for interrupted runs or iterative improvement.',
  },
  'configure-preset': {
    title: 'Training presets',
    content: 'Save and load training configurations. Create presets for common setups to avoid re-entering parameters each time.',
  },
  'train-start': {
    title: 'Start training',
    content: 'Launch the training job. You\'ll see live progress with loss curves, step counts, and estimated time remaining.',
    tip: 'You can Ctrl+Enter to start training from anywhere on the page.',
  },
  'train-progress': {
    title: 'Live training progress',
    content: 'Watch the loss curve in real-time. A decreasing loss means the model is learning. The curve should trend downward — if it plateaus or rises, try fewer epochs or a lower learning rate.',
  },
  'train-controls': {
    title: 'Training controls',
    content: 'Pause: temporarily stop training (can resume). Stop: cancel the training job entirely. The current checkpoint is saved.',
  },
  'results-checkpoint': {
    title: 'Checkpoints',
    content: 'Each training run saves a checkpoint — a snapshot of the trained model. Load a checkpoint to use it with the agent, or delete old ones to save disk space.',
    tip: 'The "Best" badge marks the checkpoint with the lowest loss — that\'s usually the one you want.',
  },
  'results-load': {
    title: 'Load checkpoint',
    content: 'Load a checkpoint into the agent\'s active model. The agent will use this trained model for all future conversations until you load a different one.',
  },
  'results-test': {
    title: 'Test model',
    content: 'Send a test prompt to the trained model to see how it performs. Compare outputs across different checkpoints to find the best one.',
  },
}

interface StepContextProps {
  hintKey: string
  children: React.ReactNode
  className?: string
}

export function StepContext({ hintKey, children, className }: StepContextProps) {
  const [isOpen, setIsOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const hint = STEP_HINTS[hintKey]

  useEffect(() => {
    if (!isOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  if (!hint) return <>{children}</>

  return (
    <div ref={ref} className={cn('relative inline-flex', className)}>
      <div className="flex items-center gap-1.5">
        {children}
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="inline-flex items-center justify-center w-4 h-4 rounded-full text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/50 transition-colors"
          aria-label={`Help: ${hint.title}`}
          aria-expanded={isOpen}
        >
          <IconInfo className="w-3 h-3" />
        </button>
      </div>
      {isOpen && (
        <div className="absolute z-40 top-full left-0 mt-2 w-72 rounded-lg border border-border/50 bg-card shadow-lg p-4 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-xs font-medium text-foreground">{hint.title}</h3>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="shrink-0 w-4 h-4 flex items-center justify-center rounded text-muted-foreground/50 hover:text-foreground transition-colors"
              aria-label="Close help"
            >
              <IconX className="w-3 h-3" />
            </button>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed">{hint.content}</p>
          {hint.tip && (
            <div className="rounded-md bg-primary/5 border border-primary/10 px-3 py-2">
              <p className="text-[11px] text-primary/80 leading-relaxed">
                <span className="font-medium">Tip:</span> {hint.tip}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export { STEP_HINTS }
