'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingTipsProps {
  checkpoints: Checkpoint[]
  isTraining?: boolean
  hasDataset?: boolean
}

interface Tip {
  id: string
  title: string
  desc: string
  variant: 'info' | 'success' | 'warning'
}

function generateTips(checkpoints: Checkpoint[], isTraining: boolean, hasDataset: boolean): Tip[] {
  const tips: Tip[] = []
  const withLoss = checkpoints.filter(c => c.loss != null && c.loss > 0)

  if (checkpoints.length === 0 && !isTraining) {
    tips.push({
      id: 'get-started',
      title: 'Get started',
      desc: 'Select a dataset on the Train tab and click Start training to create your first model.',
      variant: 'info',
    })
  }

  if (checkpoints.length === 1 && withLoss.length === 1) {
    const loss = withLoss[0].loss!
    tips.push({
      id: 'first-checkpoint',
      title: 'First checkpoint saved',
      desc: `Loss: ${loss.toFixed(3)}. Train more epochs or try a different dataset to improve.`,
      variant: 'success',
    })
  }

  if (withLoss.length >= 2) {
    const sorted = [...withLoss].sort((a, b) => (a.loss ?? Infinity) - (b.loss ?? Infinity))
    const best = sorted[0]
    const worst = sorted[sorted.length - 1]
    if (best.loss != null && worst.loss != null && worst.loss - best.loss > 1) {
      tips.push({
        id: 'loss-spread',
        title: 'Large loss spread detected',
        desc: `Best: ${best.loss.toFixed(3)} vs Worst: ${worst.loss.toFixed(3)}. Consider using early stopping to avoid overfitting.`,
        variant: 'warning',
      })
    }
  }

  const overfit = checkpoints.filter(c => c.verdict === 'overfit')
  if (overfit.length > 0) {
    tips.push({
      id: 'overfit-warning',
      title: 'Overfitting detected',
      desc: `${overfit.length} checkpoint(s) show overfitting. Try fewer epochs, more data, or a smaller model.`,
      variant: 'warning',
    })
  }

  const loaded = checkpoints.filter(c => c.is_loaded)
  if (loaded.length > 1) {
    tips.push({
      id: 'multiple-loaded',
      title: 'Multiple models loaded',
      desc: `${loaded.length} checkpoints are loaded. Unload unused ones to free memory.`,
      variant: 'warning',
    })
  }

  if (isTraining) {
    tips.push({
      id: 'training-active',
      title: 'Training in progress',
      desc: 'Monitor the loss curve. Loss should decrease steadily. If it plateaus, try adjusting the learning rate.',
      variant: 'info',
    })
  }

  if (hasDataset && checkpoints.length === 0 && !isTraining) {
    tips.push({
      id: 'ready-to-train',
      title: 'Ready to train',
      desc: 'Your dataset is selected. Click Start training on the Train tab to begin.',
      variant: 'success',
    })
  }

  const smallDataset = checkpoints.filter(c => c.training_dataset && c.epochs_trained != null && c.epochs_trained > 20)
  if (smallDataset.length > 0) {
    tips.push({
      id: 'many-epochs',
      title: 'Many epochs detected',
      desc: 'Some checkpoints trained for 20+ epochs. This may indicate overfitting on small datasets.',
      variant: 'warning',
    })
  }

  return tips.slice(0, 3)
}

const variantStyles: Record<string, string> = {
  info: 'bg-primary/5 border-primary/20',
  success: 'bg-success/5 border-success/20',
  warning: 'bg-warning/5 border-warning/20',
}

const badgeStyles: Record<string, string> = {
  info: 'bg-primary/10 text-primary border-primary/20',
  success: 'bg-success/10 text-success border-success/20',
  warning: 'bg-warning/10 text-warning border-warning/20',
}

export function TrainingTips({ checkpoints, isTraining = false, hasDataset = false }: TrainingTipsProps) {
  const tips = useMemo(() => generateTips(checkpoints, isTraining, hasDataset), [checkpoints, isTraining, hasDataset])

  if (tips.length === 0) return null

  return (
    <Card data-testid="training-tips">
      <CardHeader className="py-3">
        <CardTitle className="text-base">Tips</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {tips.map(tip => (
          <div key={tip.id} className={`rounded border p-2.5 space-y-0.5 ${variantStyles[tip.variant]}`}>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={`text-[9px] px-1 py-0 h-3.5 ${badgeStyles[tip.variant]}`}>
                {tip.variant}
              </Badge>
              <span className="text-[11px] font-medium">{tip.title}</span>
            </div>
            <p className="text-[10px] text-muted-foreground leading-relaxed">{tip.desc}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
