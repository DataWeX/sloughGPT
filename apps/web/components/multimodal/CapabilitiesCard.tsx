'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/tags'
import { StatCard, KpiGrid } from '@/components/ui'
import type { MultimodalCapabilities } from '@/lib/multimodal-controller'

interface CapabilitiesCardProps {
  caps: MultimodalCapabilities | null
}

export default function CapabilitiesCard({ caps }: CapabilitiesCardProps) {
  const capList = caps ? [
    { label: 'Speech-to-text', ok: caps.speech_to_text },
    { label: 'Image captioning', ok: caps.image_caption },
    { label: 'Vision model', ok: !!caps.vision_model },
    { label: 'Speech model', ok: !!caps.speech_model },
    { label: 'Trained', ok: caps.trained },
  ] : []

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Capabilities</CardTitle></CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2 mb-4">
          {capList.map(c => (
            <Badge key={c.label} label={c.label} variant={c.ok ? 'success' : 'warning'} size="sm" />
          ))}
        </div>
        <KpiGrid columns={4}>
          <StatCard label="Images learned" value={caps?.images_learned ?? 0} />
          <StatCard label="Memory" value={`${caps?.replay_buffer_size ?? 0} items`} />
          <StatCard label="Learning method" value={caps?.learning_method || '—'} />
          <StatCard label="Status" value={caps?.status || '—'} />
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
