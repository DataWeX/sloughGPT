'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle, Button, Badge, Input } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { apiGet } from '@/lib/http-client'
import { AutoTrainStatusCard } from '@/components/auto-train/AutoTrainStatusCard'
import { AutoTrainConfigCard } from '@/components/auto-train/AutoTrainConfigCard'
import { AutoTrainHistoryCard } from '@/components/auto-train/AutoTrainHistoryCard'
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
      const [statusData, statsData] = await Promise.all([
        settingsController.getAutoTrainSettingsStatus(),
        apiGet('/mobile/train/stats'),
      ])
      if (statusData) {
        setStatus(statusData as unknown as AutoTrainStatus)
        setThreshold((statusData as any).threshold || 10)
      }
      if (statsData) {
        setStats(statsData as unknown as TrainingStats)
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
      await settingsController.updateAutoTrainSettingsConfig({ threshold, interval_s: intervalS })
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
          <AutoTrainStatusCard status={status} stats={stats} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <AutoTrainConfigCard
              threshold={status?.threshold}
              intervalS={120}
              onSave={async (t, i) => {
                await settingsController.updateAutoTrainSettingsConfig({ threshold: t, interval_s: i })
                fetchData()
              }}
            />
            <AutoTrainHistoryCard lastTrain={status?.last_train ?? null} />
          </div>
        </>
      )}
    </PageContainer>
  )
}
