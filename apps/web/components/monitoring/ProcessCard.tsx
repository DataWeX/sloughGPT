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
            value={sys.cpu_percent != null ? sys.cpu_percent + '%' : '...'}
            numeric
          />
          <StatCard
            label="Memory"
            value={sys.memory_percent != null ? sys.memory_percent + '%' : '...'}
            numeric
          />
          {hasProcessData && (
            <>
              <StatCard
                label="Open files"
                value={sys.open_files != null ? sys.open_files : '...'}
                numeric
              />
              <StatCard
                label="Threads"
                value={sys.threads != null ? sys.threads : '...'}
                numeric
              />
            </>
          )}
          {sys.gc_gen0 != null && (
            <>
              <StatCard label="GC Gen 0" value={sys.gc_gen0} numeric />
              <StatCard label="GC Gen 1" value={sys.gc_gen1 ?? 0} numeric />
              <StatCard label="GC Gen 2" value={sys.gc_gen2 ?? 0} numeric />
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
                value={detailed.gpu.vram_gb ? detailed.gpu.vram_gb + ' GB' : '...'}
                numeric
              />
            </>
          )}
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
