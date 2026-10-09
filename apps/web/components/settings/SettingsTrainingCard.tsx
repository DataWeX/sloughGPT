'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle, Input, Switch } from '@sloughgpt/strui'

interface SettingsTrainingCardProps {
  preferredModel: string
  preferredMethod: string
  maxCheckpoints: number
  autoTrainThreshold: number
  autoTrain: boolean
  enableTracking: boolean
  onPreferredModelChange: (model: string) => void
  onPreferredMethodChange: (method: string) => void
  onMaxCheckpointsChange: (n: number) => void
  onAutoTrainThresholdChange: (threshold: number) => void
  onAutoTrainChange: (enabled: boolean) => void
  onEnableTrackingChange: (enabled: boolean) => void
}

export function SettingsTrainingCard({
  preferredModel,
  preferredMethod,
  maxCheckpoints,
  autoTrainThreshold,
  autoTrain,
  enableTracking,
  onPreferredModelChange,
  onPreferredMethodChange,
  onMaxCheckpointsChange,
  onAutoTrainThresholdChange,
  onAutoTrainChange,
  onEnableTrackingChange,
}: SettingsTrainingCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training</CardTitle>
        <CardDescription>Default training parameters</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Preferred model</label>
            <Input
              value={preferredModel}
              onChange={(e) => onPreferredModelChange(e.target.value)}
              placeholder="e.g. slo-1.6b"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Preferred method</label>
            <Input
              value={preferredMethod}
              onChange={(e) => onPreferredMethodChange(e.target.value)}
              placeholder="e.g. finetune"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Max checkpoints</label>
            <Input
              type="number"
              min={1}
              max={50}
              value={maxCheckpoints}
              onChange={(e) => onMaxCheckpointsChange(parseInt(e.target.value) || 10)}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Auto-train threshold</label>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={autoTrainThreshold}
              onChange={(e) => onAutoTrainThresholdChange(parseFloat(e.target.value) || 0.8)}
            />
          </div>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Auto-train</p>
            <p className="text-xs text-muted-foreground">Automatically train when data quality exceeds threshold</p>
          </div>
          <Switch
            checked={autoTrain}
            onCheckedChange={onAutoTrainChange}
          />
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Enable outcome tracking</p>
            <p className="text-xs text-muted-foreground">Record training runs for adaptive learning</p>
          </div>
          <Switch
            checked={enableTracking}
            onCheckedChange={onEnableTrackingChange}
          />
        </div>
      </CardContent>
    </Card>
  )
}
