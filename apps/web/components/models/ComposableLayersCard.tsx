'use client'

import { memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, IconBrain, IconModels, IconTraining } from '@sloughgpt/strui'
import type { ReactNode } from 'react'
import type { Soul, Checkpoint } from '@/lib/souls-controller'

interface ComposableLayersCardProps {
  modelsCount: number
  soulsCount: number
  checkpoints: Checkpoint[]
}

export default memo(function ComposableLayersCard({ modelsCount, soulsCount, checkpoints }: ComposableLayersCardProps) {
  const layers: { title: string; desc: string; icon: ReactNode; count: number }[] = [
    { title: 'Base Models', desc: 'Load any HuggingFace model as the foundation layer', icon: <IconModels className="h-4 w-4 text-primary" />, count: modelsCount },
    { title: 'Personalities', desc: 'Soul profiles that wrap the model with traits & voice', icon: <IconBrain className="h-4 w-4 text-accent" />, count: soulsCount },
    { title: 'Adapters', desc: 'LoRA/DoRA fine-tuned adapters that stack on any base', icon: <IconBrain className="h-4 w-4 text-success" />, count: checkpoints.filter((c: Checkpoint) => c.soul).length },
    { title: 'Checkpoints', desc: 'Trained checkpoints that persist a model+personality snapshot', icon: <IconTraining className="h-4 w-4 text-warning" />, count: checkpoints.length },
  ]

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Composable Layers</CardTitle></CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {layers.map(layer => (
            <div key={layer.title} className="rounded-lg border border-border/60 p-3">
              <div className="flex items-center gap-2 mb-1">
                {layer.icon}
                <span className="text-sm font-medium">{layer.title}</span>
              </div>
              <p className="text-[11px] text-muted-foreground mb-2">{layer.desc}</p>
              <span className="text-[10px] text-muted-foreground/60">{layer.count} available</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
})
