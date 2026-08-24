'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { memoryController, type MemoryConfigResult } from '@/lib/memory-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

export function MemorySettingsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [config, setConfig] = useState<MemoryConfigResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [toggling, setToggling] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      const c = await memoryController.getConfig()
      setConfig(c)
    } catch {
      addToast('Could not load memory config', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchConfig() }, [fetchConfig])

  const handleToggle = async (enabled: boolean) => {
    setToggling(true)
    try {
      const updated = await memoryController.setEnabled(enabled)
      setConfig(updated)
      addToast(enabled ? 'Auto-memory enabled' : 'Auto-memory disabled', 'success')
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Could not update memory settings'), 'error')
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
        <div className="text-xs text-muted-foreground">
          Store: {config.store_path}
        </div>
        <Button
          size="sm"
          variant="outline"
          className="h-7 text-xs"
          onClick={() => memoryController.consolidate().then(() => addToast('Consolidation triggered', 'success')).catch(e => addToast(extractErrorMessage(e, 'Could not consolidation'), 'error'))}
        >
          Run Consolidation Now
        </Button>
      </CardContent>
    </Card>
  )
}
