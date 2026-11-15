'use client'
import { notify } from '@/lib/notify'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { memoryController, type MemoryConfigResult } from '@/lib/memory-controller'
import { extractErrorMessage } from '@/lib/error-utils'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export function MemorySettingsCard() {
  const [config, setConfig] = useState<MemoryConfigResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [toggling, setToggling] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      const c = await memoryController.getConfig()
      setConfig(c)
    } catch {
      notify.toast('Could not load memory config', { type: 'error' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const c = await memoryController.getConfig()
        if (active) setConfig(c)
      } catch {
        if (active) notify.toast('Could not load memory config', { type: 'error' })
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [])

  const handleToggle = async (enabled: boolean) => {
    setToggling(true)
    try {
      const updated = await memoryController.setEnabled(enabled)
      setConfig(updated)
      notify.toast(enabled ? 'Auto-memory enabled' : 'Auto-memory disabled', { type: 'success' })
    } catch (e: unknown) {
      notify.toast(extractErrorMessage(e, 'Could not update memory settings'), { type: 'error' })
    } finally {
      setToggling(false)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (!config) return null

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Auto-Memory</CardTitle>
            <CardDescription>Remember facts from conversations automatically</CardDescription>
          </div>
          <Switch
            checked={config.enabled}
            onCheckedChange={handleToggle}
            disabled={toggling}
            aria-label="Toggle auto-memory"
          />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Max facts</span>
            <p className="font-mono text-xs">{config.max_facts}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Min characters</span>
            <p className="font-mono text-xs">{config.min_chars}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Archive retention</span>
            <p className="font-mono text-xs">{config.archive_retention_days} days</p>
          </div>
          <div>
            <span className="text-muted-foreground">Consolidation</span>
            <p className="font-mono text-xs">{config.consolidation_threshold}</p>
          </div>
        </div>
        <div className="text-xs text-muted-foreground">Store: {config.store_path}</div>
        <Button
          size="sm"
          variant="outline"
          className="h-7 text-xs"
          onClick={() =>
            memoryController
              .consolidate()
              .then(() => notify.toast('Consolidation triggered', { type: 'success' }))
              .catch((e) =>
                notify.toast(extractErrorMessage(e, 'Could not consolidation'), { type: 'error' }),
              )
          }
        >
          Run Consolidation Now
        </Button>
      </CardContent>
    </Card>
  )
}
