'use client'

/** Training config card. */
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Switch,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsTraining({ wb }: { wb: SettingsWorkbench }) {
  const { settings, updateSettings } = wb

  return (
    <>
      {/* Training config */}
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
                value={settings.trainingPreferredModel || ''}
                onChange={(e) => updateSettings({ trainingPreferredModel: e.target.value })}
                placeholder="e.g. slo-1.6b"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Preferred method</label>
              <Input
                value={settings.trainingPreferredMethod || ''}
                onChange={(e) => updateSettings({ trainingPreferredMethod: e.target.value })}
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
                value={settings.trainingMaxCheckpoints ?? 10}
                onChange={(e) =>
                  updateSettings({ trainingMaxCheckpoints: parseInt(e.target.value) || 10 })
                }
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Auto-train threshold</label>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={settings.trainingAutoTrainThreshold ?? 0.8}
                onChange={(e) =>
                  updateSettings({ trainingAutoTrainThreshold: parseFloat(e.target.value) || 0.8 })
                }
              />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Auto-train</p>
              <p className="text-xs text-muted-foreground">
                Automatically train when data quality exceeds threshold
              </p>
            </div>
            <Switch
              checked={settings.trainingAutoTrain ?? false}
              onCheckedChange={(checked) => updateSettings({ trainingAutoTrain: checked })}
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Enable outcome tracking</p>
              <p className="text-xs text-muted-foreground">
                Record training runs for adaptive learning
              </p>
            </div>
            <Switch
              checked={settings.trainingEnableTracking ?? true}
              onCheckedChange={(checked) => updateSettings({ trainingEnableTracking: checked })}
            />
          </div>
        </CardContent>
      </Card>
    </>
  )
}
