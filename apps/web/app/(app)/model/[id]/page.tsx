'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'
import { modelDisplayName } from '@/lib/inference-display'
import {
  ArrowLeft, Activity, Cpu, Zap, AlertTriangle, CheckCircle,
  Clock, BarChart3, Settings, ExternalLink
} from 'lucide-react'

interface ModelDetail {
  model_id: string
  status: string
  device: string
  is_default: boolean
  circuit_breaker: string
  warmup_completed: boolean
  warmup_error: string | null
  total_requests?: number
  total_tokens?: number
  avg_latency_ms?: number
  p95_latency_ms?: number
  p99_latency_ms?: number
  error_count?: number
  last_request_at?: string
  queue_depth_total?: number
  queue_served?: number
  queue_avg_wait_ms?: number
  queue_max_wait_ms?: number
  tokens_per_sec?: number
}

const STATUS_CONFIG: Record<string, { color: string; icon: typeof CheckCircle }> = {
  ready: { color: 'text-green-500', icon: CheckCircle },
  degraded: { color: 'text-yellow-500', icon: AlertTriangle },
  error: { color: 'text-red-500', icon: AlertTriangle },
  loading: { color: 'text-blue-500', icon: Activity },
  unloaded: { color: 'text-muted-foreground', icon: Clock },
}

export default function ModelDetailPage() {
  const params = useParams()
  const router = useRouter()
  const modelId = params.id as string
  const addToast = useToastStore(s => s.addToast)

  const [model, setModel] = useState<ModelDetail | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchModel = useCallback(async () => {
    setLoading(true)
    try {
      const data = await apiGet<ModelDetail>(`/registry/models/${modelId}`)
      setModel(data)
    } catch {
      addToast('Failed to load model details', 'error')
    } finally {
      setLoading(false)
    }
  }, [modelId, addToast])

  useEffect(() => { fetchModel() }, [fetchModel])

  const statusCfg = model ? STATUS_CONFIG[model.status] ?? STATUS_CONFIG.unloaded : null
  const StatusIcon = statusCfg?.icon ?? Clock

  return (
    <PageContainer
      title={model ? modelDisplayName(model.model_id) : 'Model Details'}
      subtitle={model ? `Status: ${model.status}` : undefined}
      loading={loading}
      loadingCards={4}
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => router.push('/models')}>
            <ArrowLeft className="h-4 w-4 mr-1" /> Back
          </Button>
          <Button variant="outline" size="sm" onClick={fetchModel}>
            <IconRefresh className="h-4 w-4" />
          </Button>
        </div>
      }
    >
      {!loading && model && (
        <div className="space-y-6">
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <StatusIcon className={`h-6 w-6 ${statusCfg?.color}`} />
                <div className="flex-1">
                  <div className="font-medium text-lg">{modelDisplayName(model.model_id)}</div>
                  <div className="text-sm text-muted-foreground">
                    {model.device} {model.is_default ? '(default)' : ''}
                  </div>
                </div>
                <Badge variant={model.status === 'ready' ? 'default' : 'secondary'}>
                  {model.status}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard icon={Zap} label="Total Requests" value={model.total_requests ?? 0} />
            <MetricCard icon={BarChart3} label="Total Tokens" value={model.total_tokens ?? 0} />
            <MetricCard icon={Activity} label="Avg Latency" value={model.avg_latency_ms ? `${model.avg_latency_ms.toFixed(1)}ms` : '—'} />
            <MetricCard icon={Cpu} label="Tokens/sec" value={model.tokens_per_sec ? model.tokens_per_sec.toFixed(1) : '—'} />
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4" /> Latency Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">{model.p95_latency_ms ? `${model.p95_latency_ms.toFixed(1)}` : '—'}</div>
                  <div className="text-xs text-muted-foreground">P95 (ms)</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{model.p99_latency_ms ? `${model.p99_latency_ms.toFixed(1)}` : '—'}</div>
                  <div className="text-xs text-muted-foreground">P99 (ms)</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-500">{model.error_count ?? 0}</div>
                  <div className="text-xs text-muted-foreground">Errors</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Clock className="h-4 w-4" /> Request Queue
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <Row label="Depth" value={model.queue_depth_total ?? 0} />
                <Row label="Served" value={model.queue_served ?? 0} />
                <Row label="Avg Wait" value={model.queue_avg_wait_ms ? `${model.queue_avg_wait_ms.toFixed(1)}ms` : '—'} />
                <Row label="Max Wait" value={model.queue_max_wait_ms ? `${model.queue_max_wait_ms.toFixed(1)}ms` : '—'} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Settings className="h-4 w-4" /> Circuit Breaker
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <Row label="State" value={model.circuit_breaker} />
                <Row label="Warmup" value={model.warmup_completed ? 'Complete' : 'Pending'} />
                {model.warmup_error && (
                  <div className="text-xs text-red-500 mt-2">{model.warmup_error}</div>
                )}
                {model.last_request_at && (
                  <Row label="Last Request" value={new Date(model.last_request_at).toLocaleString()} />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" onClick={() => router.push('/training/queue')}>
                  Training Queue
                </Button>
                <Button variant="outline" size="sm" onClick={() => router.push('/training/trends')}>
                  Training Trends
                </Button>
                <Button variant="outline" size="sm" onClick={() => router.push('/training/insights')}>
                  Adaptive Insights
                </Button>
                <Button variant="outline" size="sm" onClick={() => window.open(`https://huggingface.co/${model.model_id}`, '_blank')}>
                  <ExternalLink className="h-3 w-3 mr-1" /> HuggingFace
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}

function MetricCard({ icon: Icon, label, value }: { icon: typeof Zap; label: string; value: number | string }) {
  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <Icon className="h-3 w-3" /> {label}
        </div>
        <div className="text-xl font-bold">{value}</div>
      </CardContent>
    </Card>
  )
}

function Row({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}
