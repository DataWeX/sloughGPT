'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

interface TrainingOnboardingProps {
  hasCheckpoints: boolean
  onStartTraining?: () => void
  onImportData?: () => void
}

const steps = [
  { num: 1, title: 'Add training data', desc: 'Import a dataset or paste your own text' },
  { num: 2, title: 'Configure & start', desc: 'Pick a model type, set epochs, and click Start' },
  { num: 3, title: 'Monitor progress', desc: 'Watch the loss curve and track improvement' },
  { num: 4, title: 'Use your model', desc: 'Load the best checkpoint and try it in chat' },
]

export function TrainingOnboarding({ hasCheckpoints, onStartTraining, onImportData }: TrainingOnboardingProps) {
  if (hasCheckpoints) return null

  return (
    <Card data-testid="training-onboarding" className="border-dashed">
      <CardHeader className="py-4">
        <CardTitle className="text-base">Welcome to Training</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Train your own AI model from scratch or fine-tune an existing one. No coding required.
        </p>

        <div className="grid grid-cols-2 gap-3">
          {steps.map(step => (
            <div key={step.num} className="flex gap-2.5">
              <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-medium">
                {step.num}
              </div>
              <div className="space-y-0.5">
                <p className="text-[11px] font-medium">{step.title}</p>
                <p className="text-[10px] text-muted-foreground/70">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-2 pt-1">
          {onImportData && (
            <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={onImportData}>
              Import data
            </Button>
          )}
          {onStartTraining && (
            <Button size="sm" className="h-7 text-[11px]" onClick={onStartTraining}>
              Start training
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
