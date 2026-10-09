'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { modelController, type ProviderDiagnostics, type StartupProgress } from '@/lib/model-controller'

const POLL_MS = 2000
const DONE_PHASES = new Set(['ready', 'running'])

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={cn("inline-block w-1.5 h-1.5 rounded-full shrink-0", ok ? "bg-success" : "bg-destructive")} />
}

function KvRow({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-2 text-[10px]">
      <span className="text-muted-foreground/60 shrink-0 w-28">{label}</span>
      <span className={cn("text-foreground truncate", mono && "font-mono")}>{value ?? '—'}</span>
    </div>
  )
}

function StartupBar({ progress }: { progress: StartupProgress }) {
  const pct = progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : 0
  const done = DONE_PHASES.has(progress.phase)
  return (
    <div className="space-y-1.5 rounded-md border border-border/40 bg-muted/30 p-3">
      <div className="flex items-center justify-between text-xs font-medium text-muted-foreground uppercase tracking-wider">
        <span>Startup</span>
        <span className="font-mono">{done ? 'complete' : `${progress.step}/${progress.total}`}</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-500", done ? "bg-success" : "bg-primary")} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-muted-foreground truncate">{progress.message}</p>
    </div>
  )
}

export default memo(function ProviderDiagnosticsCard() {
  const [data, setData] = useState<ProviderDiagnostics | null>(null)
  const [startup, setStartup] = useState<StartupProgress | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

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

  const pollStartup = useCallback(async () => {
    const p = await modelController.getStartupProgress()
    if (p) {
      setStartup(p)
      if (DONE_PHASES.has(p.phase) && pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    void refresh()
    void pollStartup()
    pollRef.current = setInterval(() => void pollStartup(), POLL_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [refresh, pollStartup])

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
        <Button variant="ghost" size="sm" onClick={() => void refresh()} disabled={loading} className="h-7 text-[10px]">
          {loading ? 'Loading...' : 'Refresh'}
        </Button>
      </CardHeader>
      <CardContent className="space-y-2">
        {error && <p className="text-[10px] text-destructive">{error}</p>}

        {startup && <StartupBar progress={startup} />}

        {modelState && (
          <div className="space-y-1 rounded-md border border-border/40 bg-muted/30 p-2.5">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Model State</div>
            <KvRow label="model" value={modelState.model} mono />
            <KvRow label="model_type" value={modelState.model_type} mono />
            <KvRow label="tokenizer" value={modelState.tokenizer} mono />
            <KvRow label="provider (state)" value={modelState.provider} mono />
          </div>
        )}

        <div className="space-y-1 rounded-md border border-border/40 bg-muted/30 p-2.5">
          <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-0.5">Default Router</div>
          {defaultRouter ? (
            <>
              <KvRow label="text_provider" value={defaultRouter.text_provider ?? 'NONE — chat broken'} mono />
              <KvRow label="processors" value={defaultRouter.processors?.join(', ') ?? '—'} />
            </>
          ) : (
            <p className="text-[10px] text-destructive">No default router registered</p>
          )}
        </div>

        {textProvider && (
          <div className="space-y-1 rounded-md border border-border/40 bg-muted/30 p-2.5">
            <div className="flex items-center gap-1.5 mb-0.5">
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

        <div className="space-y-0.5">
          <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">All Providers ({Object.keys(providers).length})</div>
          {Object.entries(providers).map(([name, info]) => (
            <div key={name} className="flex items-center gap-1.5 text-[10px] py-0.5">
              <StatusDot ok={!!info.model_id || name === 'default'} />
              <span className="font-mono text-foreground w-28 truncate">{name}</span>
              <span className="text-muted-foreground/60">{info.type}</span>
              {info.text_provider && <span className="text-muted-foreground/60">→ {info.text_provider}</span>}
            </div>
          ))}
          {Object.keys(providers).length === 0 && (
            <p className="text-[10px] text-muted-foreground/60">No providers registered</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
})
