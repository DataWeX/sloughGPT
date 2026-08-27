'use client'

import { memo, useCallback, useEffect, useState } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button, Chip } from '@sloughgpt/strui'
import { modelController, type ProviderDiagnostics } from '@/lib/model-controller'

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={cn("inline-block w-2 h-2 rounded-full shrink-0", ok ? "bg-success" : "bg-destructive")} />
}

function KvRow({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-2 text-xs">
      <span className="text-muted-foreground shrink-0 w-28">{label}</span>
      <span className={cn("text-foreground truncate", mono && "font-mono")}>{value ?? '—'}</span>
    </div>
  )
}

export default memo(function ProviderDiagnosticsCard() {
  const [data, setData] = useState<ProviderDiagnostics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await modelController.debugProviders()
      setData(result)
      if (!result) setError('No response from server')
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const providers = data?.providers ?? {}
  const modelState = data?.model_state
  const defaultName = data?.default_provider
  const defaultRouter = defaultName ? providers[defaultName] : null
  const textProviderName = defaultRouter?.text_provider ?? null
  const textProvider = textProviderName ? providers[textProviderName] : null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base">Provider Chain</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void refresh()}
          disabled={loading}
          className="h-8 text-xs"
        >
          {loading ? 'Loading...' : 'Refresh'}
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && (
          <p className="text-xs text-destructive">{error}</p>
        )}

        {/* Model state */}
        {modelState && (
          <div className="space-y-1.5 rounded-md border border-border/40 bg-muted/30 p-3">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Model State</div>
            <KvRow label="model" value={modelState.model} mono />
            <KvRow label="model_type" value={modelState.model_type} mono />
            <KvRow label="tokenizer" value={modelState.tokenizer} mono />
            <KvRow label="provider (state)" value={modelState.provider} mono />
          </div>
        )}

        {/* Default router */}
        <div className="space-y-1.5 rounded-md border border-border/40 bg-muted/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Default Router</div>
            {defaultRouter && <Chip label={defaultRouter.type} />}
          </div>
          {defaultRouter ? (
            <>
              <KvRow label="text_provider" value={defaultRouter.text_provider ?? 'NONE — chat broken'} mono />
              <KvRow label="processors" value={defaultRouter.processors?.join(', ') ?? '—'} />
            </>
          ) : (
            <p className="text-xs text-destructive">No default router registered</p>
          )}
        </div>

        {/* Text provider detail */}
        {textProvider && (
          <div className="space-y-1.5 rounded-md border border-border/40 bg-muted/30 p-3">
            <div className="flex items-center gap-2 mb-1">
              <StatusDot ok={!!textProvider.model_id} />
              <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Text Provider ({textProviderName})</div>
            </div>
            <KvRow label="model_id" value={textProvider.model_id} mono />
            {textProvider.server && (
              <>
                <KvRow label="server type" value={textProvider.server.type} mono />
                <KvRow label="circuit breaker" value={textProvider.server.has_circuit_breaker ? 'yes' : 'no'} />
              </>
            )}
          </div>
        )}

        {/* All registered providers */}
        <div className="space-y-1">
          <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">All Providers ({Object.keys(providers).length})</div>
          {Object.entries(providers).map(([name, info]) => (
            <div key={name} className="flex items-center gap-2 text-xs py-0.5">
              <StatusDot ok={!!info.model_id || name === 'default'} />
              <span className="font-mono text-foreground w-28 truncate">{name}</span>
              <span className="text-muted-foreground">{info.type}</span>
              {info.text_provider && <span className="text-muted-foreground">→ {info.text_provider}</span>}
            </div>
          ))}
          {Object.keys(providers).length === 0 && (
            <p className="text-xs text-muted-foreground">No providers registered</p>
          )}
        </div>

        <div className="text-[10px] text-muted-foreground pt-1">
          Startup phase: <span className="font-mono">{data?.startup_phase ?? '—'}</span>
        </div>
      </CardContent>
    </Card>
  )
})
