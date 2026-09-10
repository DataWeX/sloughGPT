'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle, Button, Badge, Input } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { Zap, Play, Pause, Settings, Clock, Database, AlertCircle, CheckCircle } from 'lucide-react'

interface AutoTrainStatus {
  enabled: boolean
  threshold: number
  pending_count: number
  last_train: {
    started_at: string
    completed_at: string | null
    pairs_used: number
    checkpoint: string
  } | null
}

interface TrainingStats {
  total: number
  pending: number
  synced: number
  used: number
  by_quality: Record<string, number>
}

export default function AutoTrainPage() {
  const [status, setStatus] = useState<AutoTrainStatus | null>(null)
  const [stats, setStats] = useState<TrainingStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState(false)
  const [threshold, setThreshold] = useState(10)
  const [intervalS, setIntervalS] = useState(120)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [statusResp, statsResp] = await Promise.all([
        fetch('/settings/training/auto-train/status').then(r => r.json()),
        fetch('/mobile/train/stats').then(r => r.json()),
      ])
      if (statusResp.data) {
        setStatus(statusResp.data)
        setThreshold(statusResp.data.threshold || 10)
      }
      if (statsResp.data) {
        setStats(statsResp.data)
      }
    } catch (err) {
      console.error('Failed to fetch auto-train data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  const handleUpdateConfig = async () => {
    setUpdating(true)
    try {
      await fetch(`/settings/training/auto-train/config?threshold=${threshold}&interval_s=${intervalS}`, { method: 'PATCH' })
      fetchData()
    } catch (err) {
      console.error('Failed to update config:', err)
    } finally {
      setUpdating(false)
    }
  }

  const formatDate = (ts: string | null) => {
    if (!ts) return '-'
    try { return new Date(ts).toLocaleString() } catch { return ts }
  }

  return (
    <PageContainer title="Auto-Train">
      <AppRouteHeader left={<AppRouteHeaderLead title="Auto-Train" />} />

      <p className="text-sm text-muted-foreground mb-6">
        Automatic training from conversation pairs. When enough high-quality pairs accumulate, the system trains a LoRA adapter automatically.
      </p>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="pt-6 space-y-3">
                <div className="h-5 bg-muted rounded w-1/3" />
                <div className="h-4 bg-muted rounded w-2/3" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${status?.enabled ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'}`}>
                    {status?.enabled ? <Play className="h-5 w-5" /> : <Pause className="h-5 w-5" />}
                  </div>
                  <div>
                    <p className="text-sm font-medium">{status?.enabled ? 'Enabled' : 'Disabled'}</p>
                    <p className="text-xs text-muted-foreground">Auto-train status</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
                    <Database className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">{stats?.pending || 0} pending</p>
                    <p className="text-xs text-muted-foreground">{stats?.total || 0} total pairs</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                    <Zap className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Threshold: {status?.threshold || 10}</p>
                    <p className="text-xs text-muted-foreground">Min pairs to trigger</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Settings className="h-4 w-4" /> Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Pair Threshold</label>
                  <Input
                    type="number"
                    min={10}
                    max={10000}
                    value={threshold}
                    onChange={(e) => setThreshold(parseInt(e.target.value) || 10)}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Minimum conversation pairs before auto-training triggers
                  </p>
                </div>
                <div>
                  <label className="text-xs text-muted-foreground mb-1 block">Check Interval (seconds)</label>
                  <Input
                    type="number"
                    min={30}
                    max={3600}
                    value={intervalS}
                    onChange={(e) => setIntervalS(parseInt(e.target.value) || 120)}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    How often to check for new training data
                  </p>
                </div>
                <Button onClick={handleUpdateConfig} disabled={updating}>
                  {updating ? 'Saving...' : 'Save Configuration'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Clock className="h-4 w-4" /> Last Training Run
                </CardTitle>
              </CardHeader>
              <CardContent>
                {status?.last_train ? (
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Started:</span>
                      <span>{formatDate(status.last_train.started_at)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Completed:</span>
                      <span>{formatDate(status.last_train.completed_at)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Pairs used:</span>
                      <span>{status.last_train.pairs_used}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Checkpoint:</span>
                      <Badge variant="secondary">{status.last_train.checkpoint}</Badge>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 text-muted-foreground">
                    <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-30" />
                    <p className="text-sm">No training runs yet</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {stats && Object.keys(stats.by_quality || {}).length > 0 && (
            <Card className="mt-6">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Data Quality Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-3">
                  {Object.entries(stats.by_quality).map(([quality, count]) => (
                    <div key={quality} className="flex items-center gap-2">
                      <Badge className={
                        quality === 'good' ? 'bg-green-100 text-green-800' :
                        quality === 'bad' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }>
                        {quality}
                      </Badge>
                      <span className="text-sm font-medium">{count as number}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </PageContainer>
  )
}
