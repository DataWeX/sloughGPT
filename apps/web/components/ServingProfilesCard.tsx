'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Skeleton,
} from '@sloughgpt/strui'
import {
  profilesController,
  type ServingProfile,
  type ProfileApplyResult,
} from '@/lib/profiles-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { ConfirmDialog } from '@/components/ConfirmDialog'

const TIER_LABELS: Record<string, string> = {
  cpu_only: 'CPU Only',
  cpu_optimized: 'CPU Optimized',
  balanced: 'Balanced',
  gpu_performance: 'GPU Performance',
  training: 'Training',
}

const TIER_COLORS: Record<string, string> = {
  cpu_only: 'bg-muted-foreground/30',
  cpu_optimized: 'bg-blue-500/80',
  balanced: 'bg-primary/80',
  gpu_performance: 'bg-green-500/80',
  training: 'bg-amber-500/80',
}

export function ServingProfilesCard() {
  const addToast = useToastStore(s => s.addToast)
  const [profiles, setProfiles] = useState<ServingProfile[]>([])
  const [activeId, setActiveId] = useState<string>('balanced')
  const [loading, setLoading] = useState(true)
  const [applying, setApplying] = useState<string | null>(null)
  const [applyResult, setApplyResult] = useState<ProfileApplyResult | null>(null)
  const [confirmTarget, setConfirmTarget] = useState<ServingProfile | null>(null)

  const fetchProfiles = useCallback(async () => {
    setLoading(true)
    try {
      const [list, active] = await Promise.all([
        profilesController.list(),
        profilesController.active().catch(() => ({ active_profile_id: 'balanced', profile: null })),
      ])
      setProfiles(list)
      setActiveId(active.active_profile_id)
    } catch {
      // silent
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchProfiles() }, [fetchProfiles])

  const handleApply = async (id: string) => {
    setApplying(id)
    try {
      const result = await profilesController.apply(id)
      setApplyResult(result)
      setActiveId(id)
      addToast(`Applied "${result.profile.name}" profile`, 'success')
      if (result.requires_restart.length > 0) {
        addToast(`${result.requires_restart.length} settings require restart to take effect`, 'info')
      }
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Could not apply profile'), 'error')
    } finally {
      setApplying(null)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xs">Serving Profile</CardTitle>
            <CardDescription className="text-[10px] text-muted-foreground/60">Predefined hardware-optimized configurations</CardDescription>
          </div>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={fetchProfiles}>
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map(i => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : profiles.length === 0 ? (
          <p className="text-[10px] text-muted-foreground/60">No profiles available</p>
        ) : (
          <div className="space-y-1.5">
            {profiles.map(p => {
              const isActive = p.id === activeId
              const isApplying = applying === p.id
              return (
                <div
                  key={p.id}
                  className={`flex items-center gap-2 p-2 rounded-lg border transition-colors ${
                    isActive
                      ? 'border-primary/50 bg-primary/5'
                      : 'border-border/40 hover:border-border'
                  }`}
                >
                  <div className={`h-2 w-2 rounded-full shrink-0 ${TIER_COLORS[p.tier] ?? 'bg-muted-foreground/30'}`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-medium truncate">{p.name}</span>
                      {isActive && (
                        <Badge variant="outline" className="text-[8px] px-1 py-0 h-3.5 text-primary border-primary/40">
                          Active
                        </Badge>
                      )}
                    </div>
                    <p className="text-[9px] text-muted-foreground/60 truncate">{p.description}</p>
                    <div className="flex items-center gap-2 mt-0.5 text-[9px] text-muted-foreground/60 font-mono tabular-nums">
                      <span>{p.device}</span>
                      {p.quantize && <span>int{p.quant_bits}</span>}
                      <span>{p.inference_pool_size} slots</span>
                      {p.min_ram_gb > 0 && <span>{p.min_ram_gb}GB+</span>}
                    </div>
                  </div>
                  {!isActive && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 text-[10px] shrink-0"
                      disabled={isApplying}
                      onClick={() => setConfirmTarget(p)}
                    >
                      {isApplying ? 'Applying...' : 'Apply'}
                    </Button>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {applyResult && (
          <div className="mt-3 p-2 rounded-lg bg-muted/30 border border-border/30">
            <p className="text-[10px] font-medium mb-0.5">
              Last applied: <span className="text-primary">{applyResult.profile.name}</span>
            </p>
            <div className="text-[9px] text-muted-foreground/60 space-y-px">
              <p>Applied live: {Object.keys(applyResult.live_settings).join(', ') || 'none'}</p>
              {applyResult.requires_restart.length > 0 && (
                <p className="text-amber-500">Requires restart: {applyResult.requires_restart.join(', ')}</p>
              )}
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="justify-end py-2">
        <Badge variant="outline" className="text-[8px] font-mono px-1 py-0 h-3.5 text-muted-foreground/60 border-border/40">
          {profiles.length} profiles
        </Badge>
      </CardFooter>
      <ConfirmDialog
        open={confirmTarget !== null}
        onOpenChange={(open) => { if (!open) setConfirmTarget(null) }}
        title={`Apply "${confirmTarget?.name ?? ''}" profile?`}
        description="This will change serving configuration. Some settings require a restart to take effect."
        confirmLabel="Apply"
        destructive={false}
        onConfirm={() => {
          if (confirmTarget) handleApply(confirmTarget.id)
          setConfirmTarget(null)
        }}
      />
    </Card>
  )
}
