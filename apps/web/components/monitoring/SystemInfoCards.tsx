'use client'

import { memo, useMemo } from 'react'
import { cn, Card, CardContent } from '@sloughgpt/strui'
import type { GPUInfo, DiskUsage, SystemInfo } from '@/lib/system-controller'

export const GpuCard = memo(function GpuCard({ gpu }: { gpu?: GPUInfo }) {
  const hint = useMemo(() => {
    if (!gpu) return null
    try { return JSON.parse(gpu.memory_hint) } catch { return null }
  }, [gpu])
  if (!gpu) return null
  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">GPU</span>
      <CardContent className="p-0 space-y-px text-[10px]">
        <div className="grid grid-cols-2 gap-x-2 gap-y-px">
          <span className="text-muted-foreground/60">Backend</span>
          <span className="text-right truncate font-mono">{gpu.backend}</span>
          <span className="text-muted-foreground/60">Device</span>
          <span className="text-right truncate font-mono">{gpu.device_type}</span>
          <span className="text-muted-foreground/60">VRAM</span>
          <span className="text-right font-mono tabular-nums">{gpu.vram_gb} GB</span>
          <span className="text-muted-foreground/60">Tier</span>
          <span className="text-right font-mono">{gpu.tier}</span>
          {hint && Object.entries(hint).filter(([k]) => !['tier'].includes(k)).map(([k, v]) => (
            <div key={k} className="contents">
              <span className="text-muted-foreground/60 capitalize">{k.replace(/_/g, ' ')}</span>
              <span className="text-right text-[9px] font-mono">{typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
})

export const DiskCard = memo(function DiskCard({ disk }: { disk?: DiskUsage }) {
  if (!disk) return null
  const pct = Math.round(disk.percent)
  const color = pct > 90 ? 'bg-destructive' : pct > 75 ? 'bg-warning' : 'bg-success'
  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">Disk</span>
      <CardContent className="p-0 space-y-1.5">
        <div className="flex justify-between text-[10px] text-muted-foreground/60 font-mono tabular-nums">
          <span>{disk.used_gb.toFixed(1)} GB used</span>
          <span>{disk.total_gb.toFixed(1)} GB total</span>
        </div>
        <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
          <div className={cn('absolute inset-y-0 left-0', color, 'rounded-full transition-all duration-300')} style={{ width: `${pct}%` }} />
        </div>
        <div className="flex justify-between text-[9px] text-muted-foreground/60 font-mono tabular-nums">
          <span>{disk.free_gb.toFixed(1)} GB free</span>
          <span>{pct}%</span>
        </div>
      </CardContent>
    </Card>
  )
})

export const ServerInfoCard = memo(function ServerInfoCard({ info }: { info?: SystemInfo }) {
  if (!info) return null
  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">Server</span>
      <CardContent className="p-0 space-y-px text-[10px]">
        <div className="flex justify-between py-px"><span className="text-muted-foreground/60">Platform</span><span className="font-mono">{info.platform} {info.platform_release}</span></div>
        <div className="flex justify-between py-px"><span className="text-muted-foreground/60">Architecture</span><span className="font-mono">{info.architecture}</span></div>
        <div className="flex justify-between py-px"><span className="text-muted-foreground/60">CPU cores</span><span className="font-mono tabular-nums">{info.cpu_count}</span></div>
        <div className="flex justify-between py-px"><span className="text-muted-foreground/60">Processor</span><span className="text-[9px] max-w-[180px] text-right truncate font-mono" title={info.processor}>{info.processor || '—'}</span></div>
      </CardContent>
    </Card>
  )
})
