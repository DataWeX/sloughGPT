'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'

interface ProcessCardProps {
  detailed: {
    system?: {
      cpu_percent?: number
      memory_percent?: number
      memory_available_mb?: number
      open_files?: number
      threads?: number
      gc_gen0?: number
      gc_gen1?: number
      gc_gen2?: number
    }
    gpu?: {
      backend?: string
      device_type?: string
      vram_gb?: number
      tier?: string
    }
  } | null
}

export function ProcessCard({ detailed }: ProcessCardProps) {
  if (!detailed?.system) return null

  const sys = detailed.system
  const hasProcessData = sys.open_files != null || sys.threads != null || sys.gc_gen0 != null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Process</span>
      <CardContent className="p-0">
        <KpiGrid columns={2}>
          <StatCard
            label="CPU"
            value={sys.cpu_percent != null ? <span className="font-mono">{sys.cpu_percent}%</span> : '...'}
          />
          <StatCard
            label="Memory"
            value={sys.memory_percent != null ? <span className="font-mono">{sys.memory_percent}%</span> : '...'}
          />
          {hasProcessData && (
            <>
              <StatCard
                label="Open files"
                value={sys.open_files != null ? <span className="font-mono">{sys.open_files}</span> : '...'}
              />
              <StatCard
                label="Threads"
                value={sys.threads != null ? <span className="font-mono">{sys.threads}</span> : '...'}
              />
            </>
          )}
          {sys.gc_gen0 != null && (
            <>
              <StatCard label="GC Gen 0" value={<span className="font-mono">{sys.gc_gen0}</span>} />
              <StatCard label="GC Gen 1" value={<span className="font-mono">{sys.gc_gen1 ?? 0}</span>} />
              <StatCard label="GC Gen 2" value={<span className="font-mono">{sys.gc_gen2 ?? 0}</span>} />
            </>
          )}
          {detailed.gpu && (
            <>
              <StatCard
                label="GPU"
                value={<span className="font-mono">{detailed.gpu.backend ?? 'N/A'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${detailed.gpu.backend && detailed.gpu.backend !== 'none' ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
              />
              <StatCard
                label="VRAM"
                value={detailed.gpu.vram_gb ? <span className="font-mono">{detailed.gpu.vram_gb} GB</span> : '...'}
              />
            </>
          )}
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
