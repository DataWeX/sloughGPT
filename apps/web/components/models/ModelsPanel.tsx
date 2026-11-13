'use client'

/**
 * Models surface (v1 route map card 20261101_154054).
 *
 * Two verbatim moves compose here:
 *  - the Developer Models tab body (catalog, lineage, quantization,
 *    downloads, engine status) — byte-for-byte, same cards, same query
 *    cache wiring as `dev-models-tab.tsx` had;
 *  - the `/model/[id]` detail view, now a FoldSection driven by `?id=`
 *    (select a catalog row → /models?id=<model>). Load-on-select only,
 *    gated by the explicit `detailLoading` boolean — never content emptiness.
 */
import { useState, useEffect, useCallback } from 'react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Button,
  Skeleton,
  FoldSection,
} from '@sloughgpt/strui'
import { useRouter } from '@/vite/next-compat/navigation'
import { apiGet } from '@/lib/http-client'
import { modelDisplayName } from '@/lib/inference-display'
import { useModels } from '@/lib/cache/api-hooks'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import ModelCatalogCard from '@/components/models/ModelCatalogCard'
import ModelLineageCard from '@/components/models/ModelLineageCard'
import QuantizationCard from '@/components/models/QuantizationCard'
import DownloadsCard from '@/components/models/DownloadsCard'
import EngineStatusCard from '@/components/models/EngineStatusCard'
import {
  Activity,
  Cpu,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Settings,
  ExternalLink,
} from 'lucide-react'

/** Per-model detail payload — verbatim from the retired /model/[id] route. */
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
  ready: { color: 'text-success', icon: CheckCircle },
  degraded: { color: 'text-warning', icon: AlertTriangle },
  error: { color: 'text-destructive', icon: AlertTriangle },
  loading: { color: 'text-info', icon: Activity },
  unloaded: { color: 'text-muted-foreground', icon: Clock },
}

export function ModelsPanel({ selectedId }: { selectedId: string | null }) {
  const router = useRouter()
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useModels()
  const { healthLegacy: health } = useLiveStatus()
  const models = modelsData ?? []
  const isOnline = health !== null && health !== 'offline'
  const activeRuntimeId =
    health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const handleModelLoaded = useCallback(async () => {
    await refetchModels()
  }, [refetchModels])

  // Detail fold — explicit loading boolean, fetch only when ?id= is set.
  const [detail, setDetail] = useState<ModelDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!selectedId) {
      setDetail(null)
      setDetailError(null)
      setDetailLoading(false)
      return
    }
    setDetailLoading(true)
    setDetailError(null)
    apiGet<ModelDetail>(`/registry/models/${encodeURIComponent(selectedId)}`)
      .then((data) => {
        if (!cancelled) setDetail(data)
      })
      .catch(() => {
        if (!cancelled) setDetailError('Failed to load model details')
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const statusCfg = detail ? (STATUS_CONFIG[detail.status] ?? STATUS_CONFIG.unloaded) : null
  const StatusIcon = statusCfg?.icon ?? Clock

  return (
    <div className="space-y-3">
      <ModelCatalogCard
        models={models}
        modelsLoading={modelsLoading}
        activeRuntimeId={activeRuntimeId}
        onModelLoaded={handleModelLoaded}
      />

      {/* Model detail (folded from the retired /model/[id] route, card 20261101_154054) */}
      <FoldSection heading="Model Details" open={selectedId !== null}>
        {selectedId && (
          <div className="space-y-3" data-testid="model-details-body">
            {detailLoading && (
              <div data-testid="model-detail-loading">
                <Skeleton className="h-20 w-full" />
              </div>
            )}
            {!detailLoading && detailError && (
              <span className="text-xs text-destructive">{detailError}</span>
            )}
            {!detailLoading && detail && (
              <>
                <Card>
                  <CardContent className="py-4">
                    <div className="flex items-center gap-3">
                      <StatusIcon className={`h-6 w-6 ${statusCfg?.color}`} />
                      <div className="flex-1">
                        <div className="font-medium text-base">
                          {modelDisplayName(detail.model_id)}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {detail.device} {detail.is_default ? '(default)' : ''}
                        </div>
                      </div>
                      <Badge variant={detail.status === 'ready' ? 'default' : 'secondary'}>
                        {detail.status}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    icon={Zap}
                    label="Total Requests"
                    value={detail.total_requests ?? 0}
                  />
                  <MetricCard
                    icon={BarChart3}
                    label="Total Tokens"
                    value={detail.total_tokens ?? 0}
                  />
                  <MetricCard
                    icon={Activity}
                    label="Avg Latency"
                    value={detail.avg_latency_ms ? `${detail.avg_latency_ms.toFixed(1)}ms` : '—'}
                  />
                  <MetricCard
                    icon={Cpu}
                    label="Tokens/sec"
                    value={detail.tokens_per_sec ? detail.tokens_per_sec.toFixed(1) : '—'}
                  />
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
                        <div className="text-base font-bold">
                          {detail.p95_latency_ms ? `${detail.p95_latency_ms.toFixed(1)}` : '—'}
                        </div>
                        <div className="text-xs text-muted-foreground">P95 (ms)</div>
                      </div>
                      <div>
                        <div className="text-base font-bold">
                          {detail.p99_latency_ms ? `${detail.p99_latency_ms.toFixed(1)}` : '—'}
                        </div>
                        <div className="text-xs text-muted-foreground">P99 (ms)</div>
                      </div>
                      <div>
                        <div className="text-base font-bold text-destructive">
                          {detail.error_count ?? 0}
                        </div>
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
                      <Row label="Depth" value={detail.queue_depth_total ?? 0} />
                      <Row label="Served" value={detail.queue_served ?? 0} />
                      <Row
                        label="Avg Wait"
                        value={
                          detail.queue_avg_wait_ms
                            ? `${detail.queue_avg_wait_ms.toFixed(1)}ms`
                            : '—'
                        }
                      />
                      <Row
                        label="Max Wait"
                        value={
                          detail.queue_max_wait_ms
                            ? `${detail.queue_max_wait_ms.toFixed(1)}ms`
                            : '—'
                        }
                      />
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Settings className="h-4 w-4" /> Circuit Breaker
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <Row label="State" value={detail.circuit_breaker} />
                      <Row
                        label="Warmup"
                        value={detail.warmup_completed ? 'Complete' : 'Pending'}
                      />
                      {detail.warmup_error && (
                        <div className="text-xs text-destructive mt-2">{detail.warmup_error}</div>
                      )}
                      {detail.last_request_at && (
                        <Row
                          label="Last Request"
                          value={new Date(detail.last_request_at).toLocaleString()}
                        />
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
                      <Button variant="outline" size="sm" onClick={() => router.push('/training')}>
                        Training
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          window.open(`https://huggingface.co/${detail.model_id}`, '_blank')
                        }
                      >
                        <ExternalLink className="h-3 w-3 mr-1" /> HuggingFace
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
      </FoldSection>

      <ModelLineageCard />
      <QuantizationCard isOnline={isOnline} />
      <DownloadsCard />
      <EngineStatusCard />
    </div>
  )
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Zap
  label: string
  value: number | string
}) {
  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
          <Icon className="h-3 w-3" /> {label}
        </div>
        <div className="text-base font-bold">{value}</div>
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
