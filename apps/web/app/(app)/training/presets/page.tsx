'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle, Button, Badge } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { Zap, Check, ArrowRight, Cpu, Database, Clock } from 'lucide-react'

interface Preset {
  name: string
  description: string
  model: string
  method: string
  epochs: number
  batch_size: number
  learning_rate: number
  max_seq_length: number
  warmup_steps: number
  weight_decay: number
  use_lora: boolean
  lora_rank: number
  lora_alpha: number
  tags: string[]
}

const METHOD_COLORS: Record<string, string> = {
  finetune: 'bg-blue-100 text-blue-800',
  lora: 'bg-purple-100 text-purple-800',
  chat: 'bg-green-100 text-green-800',
  distill: 'bg-orange-100 text-orange-800',
  rlhf: 'bg-red-100 text-red-800',
}

export default function TrainingPresetsPage() {
  const [presets, setPresets] = useState<Preset[]>([])
  const [loading, setLoading] = useState(true)
  const [applying, setApplying] = useState<string | null>(null)
  const [applied, setApplied] = useState<string | null>(null)

  const fetchPresets = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await settingsController.listTrainingPresets()
      setPresets(resp.presets as unknown as Preset[])
    } catch (err) {
      console.error('Failed to fetch presets:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchPresets() }, [fetchPresets])

  const handleApply = async (name: string) => {
    setApplying(name)
    try {
      await settingsController.applyTrainingPreset(name)
      setApplied(name)
      setTimeout(() => setApplied(null), 3000)
    } catch (err) {
      console.error('Failed to apply preset:', err)
    } finally {
      setApplying(null)
    }
  }

  return (
    <PageContainer title="Training Presets">
      <AppRouteHeader left={<AppRouteHeaderLead title="Training Presets" />} />

      <p className="text-sm text-muted-foreground mb-6">
        Quick-start templates with curated hyperparameters. Select a preset to apply it to your training settings.
      </p>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="pt-6 space-y-3">
                <div className="h-5 bg-muted rounded w-1/3" />
                <div className="h-4 bg-muted rounded w-2/3" />
                <div className="h-4 bg-muted rounded w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {presets.map((preset) => (
            <Card key={preset.name} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg">{preset.name}</CardTitle>
                  <Badge className={METHOD_COLORS[preset.method] || 'bg-gray-100 text-gray-800'}>
                    {preset.method}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{preset.description}</p>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex items-center gap-1.5">
                    <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">Model:</span>
                    <span className="font-medium">{preset.model}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Database className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">Batch:</span>
                    <span className="font-medium">{preset.batch_size}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">Epochs:</span>
                    <span className="font-medium">{preset.epochs}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Zap className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">LR:</span>
                    <span className="font-medium">{preset.learning_rate}</span>
                  </div>
                </div>

                {preset.use_lora && (
                  <div className="text-xs text-muted-foreground">
                    LoRA: rank={preset.lora_rank}, alpha={preset.lora_alpha}
                  </div>
                )}

                <div className="flex flex-wrap gap-1">
                  {preset.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>

                <Button
                  size="sm"
                  className="w-full mt-2"
                  onClick={() => handleApply(preset.name)}
                  disabled={applying === preset.name}
                  variant={applied === preset.name ? 'default' : 'outline'}
                >
                  {applied === preset.name ? (
                    <><Check className="h-4 w-4 mr-1" /> Applied</>
                  ) : applying === preset.name ? (
                    'Applying...'
                  ) : (
                    <><ArrowRight className="h-4 w-4 mr-1" /> Apply Preset</>
                  )}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </PageContainer>
  )
}
