'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { ProgressBar } from '@sloughgpt/strui'
import type { GPUInfo, DiskUsage, SystemInfo } from '@/lib/system-controller'

export function GpuCard({ gpu }: { gpu?: GPUInfo }) {
  if (!gpu) return null
  let hint: Record<string, any> | null = null
  try { hint = JSON.parse(gpu.memory_hint) } catch { hint = null }
  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">GPU</span>
      <CardContent className="p-0 space-y-0.5 text-xs">
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
          <span className="text-muted-foreground">Backend</span>
          <span className="text-right truncate font-mono">{gpu.backend}</span>
          <span className="text-muted-foreground">Device</span>
          <span className="text-right truncate font-mono">{gpu.device_type}</span>
          <span className="text-muted-foreground">VRAM</span>
          <span className="text-right font-mono">{gpu.vram_gb} GB</span>
          <span className="text-muted-foreground">Tier</span>
          <span className="text-right font-mono">{gpu.tier}</span>
          {hint && Object.entries(hint).filter(([k]) => !['tier'].includes(k)).map(([k, v]) => (
            <div key={k} className="contents">
              <span className="text-muted-foreground capitalize">{k.replace(/_/g, ' ')}</span>
              <span className="text-right text-[11px] font-mono">{typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function DiskCard({ disk }: { disk?: DiskUsage }) {
  if (!disk) return null
  const pct = Math.round(disk.percent)
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Disk</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex justify-between text-xs text-muted-foreground font-mono">
          <span>{disk.used_gb.toFixed(1)} GB used</span>
          <span>{disk.total_gb.toFixed(1)} GB total</span>
        </div>
        <ProgressBar value={pct} max={100} />
        <p className="text-xs text-muted-foreground font-mono">{disk.free_gb.toFixed(1)} GB free</p>
      </CardContent>
    </Card>
  )
}

export function ServerInfoCard({ info }: { info?: SystemInfo }) {
  if (!info) return null
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Server</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex justify-between"><span className="text-muted-foreground">Platform</span><span className="font-mono">{info.platform} {info.platform_release}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Architecture</span><span className="font-mono">{info.architecture}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">CPU cores</span><span className="font-mono">{info.cpu_count}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Processor</span><span className="text-xs max-w-[200px] text-right truncate font-mono" title={info.processor}>{info.processor || '—'}</span></div>
      </CardContent>
    </Card>
  )
}
