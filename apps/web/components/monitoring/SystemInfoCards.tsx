'use client'

import { memo, useMemo } from 'react'
import { cn, Card, CardContent, Button } from '@sloughgpt/strui'
import type { GPUInfo, DiskUsage, SystemInfo, BatteryInfo } from '@/lib/system-controller'

export const GpuCard = memo(function GpuCard({ gpu }: { gpu?: GPUInfo }) {
  const hint = useMemo(() => {
    if (!gpu) return null
    try {
      return JSON.parse(gpu.memory_hint)
    } catch {
      return null
    }
  }, [gpu])
  if (!gpu) return null
  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        GPU
      </span>
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
          {hint &&
            Object.entries(hint)
              .filter(([k]) => !['tier'].includes(k))
              .map(([k, v]) => (
                <div key={k} className="contents">
                  <span className="text-muted-foreground/60 capitalize">
                    {k.replace(/_/g, ' ')}
                  </span>
                  <span className="text-right text-[9px] font-mono">
                    {typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)}
                  </span>
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
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        Disk
      </span>
      <CardContent className="p-0 space-y-1.5">
        <div className="flex justify-between text-[10px] text-muted-foreground/60 font-mono tabular-nums">
          <span>{disk.used_gb.toFixed(1)} GB used</span>
          <span>{disk.total_gb.toFixed(1)} GB total</span>
        </div>
        <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className={cn(
              'absolute inset-y-0 left-0',
              color,
              'rounded-full transition-all duration-300',
            )}
            style={{ width: `${pct}%` }}
          />
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
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        Server
      </span>
      <CardContent className="p-0 space-y-px text-[10px]">
        <div className="flex justify-between py-px">
          <span className="text-muted-foreground/60">Platform</span>
          <span className="font-mono">
            {info.platform} {info.platform_release}
          </span>
        </div>
        <div className="flex justify-between py-px">
          <span className="text-muted-foreground/60">Architecture</span>
          <span className="font-mono">{info.architecture}</span>
        </div>
        <div className="flex justify-between py-px">
          <span className="text-muted-foreground/60">CPU cores</span>
          <span className="font-mono tabular-nums">{info.cpu_count}</span>
        </div>
        <div className="flex justify-between py-px">
          <span className="text-muted-foreground/60">Processor</span>
          <span
            className="text-[9px] max-w-[180px] text-right truncate font-mono"
            title={info.processor}
          >
            {info.processor || '—'}
          </span>
        </div>
      </CardContent>
    </Card>
  )
})

const ADVICE_LABEL: Record<BatteryInfo['advice']['action'], string> = {
  unplug: 'Unplug now',
  cap_at_80: 'Cap charge at 80%',
  plug_in: 'Plug in soon',
  maintain: 'In the 20–80% sweet spot',
}

function fmtEta(min: number | null): string | null {
  if (min == null || min < 0) return null
  if (min >= 60) return `${Math.floor(min / 60)}h${String(min % 60).padStart(2, '0')}m`
  return `${min}m`
}

export const BatteryCard = memo(function BatteryCard({
  battery,
  onSetLimit,
}: {
  battery?: BatteryInfo
  onSetLimit?: (percent: number) => void
}) {
  if (!battery) return null
  const { status, control, advice } = battery
  const pct = status.level
  const color =
    status.level_band === 'low'
      ? 'bg-destructive'
      : status.level_band === 'ok'
        ? 'bg-success'
        : 'bg-warning'
  const capped = control.current_limit != null && control.current_limit < 100
  const eta = fmtEta(status.is_charging ? status.time_to_full_min : status.time_to_empty_min)
  const etaLabel = eta
    ? status.is_charging
      ? `Full in ${eta}`
      : `${eta} left`
    : status.is_charging
      ? 'Charging'
      : 'On battery'
  const canWrite = control.supported && control.writable && !!onSetLimit

  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        Battery
      </span>
      <CardContent className="p-0 space-y-1.5">
        <div className="flex justify-between text-[10px] text-muted-foreground/60 font-mono tabular-nums">
          <span>
            {pct}% · {status.name}
          </span>
          <span>{etaLabel}</span>
        </div>
        <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className={cn(
              'absolute inset-y-0 left-0',
              color,
              'rounded-full transition-all duration-300',
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex justify-between text-[9px] text-muted-foreground/60 font-mono tabular-nums">
          <span>{status.source === 'simulated' ? 'simulated' : status.health}</span>
          <span>{pct}%</span>
        </div>
        <div className="flex items-center justify-between gap-2 pt-0.5">
          <span
            className="text-[9px] text-muted-foreground/60 font-mono truncate"
            title={control.supported ? control.reason || undefined : control.reason}
          >
            {control.supported
              ? capped
                ? `Capped at ${control.current_limit}%`
                : 'No charge cap'
              : 'Charge cap unavailable'}
          </span>
          {canWrite && (
            <Button
              variant="outline"
              size="sm"
              className="text-[9px] h-6 shrink-0"
              onClick={() => onSetLimit?.(capped ? 100 : advice.limit)}
            >
              {capped ? 'Lift cap' : `Cap ${advice.limit}%`}
            </Button>
          )}
        </div>
        <div
          className="text-[9px] text-muted-foreground/70 truncate"
          title={advice.reason}
          data-testid="battery-advice"
        >
          {ADVICE_LABEL[advice.action]}
        </div>
      </CardContent>
    </Card>
  )
})
