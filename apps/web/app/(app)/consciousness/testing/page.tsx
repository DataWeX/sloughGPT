'use client'

import { useState, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const ENDPOINTS = [
  { label: 'GET /consciousness/status', method: 'GET' as const, path: '/consciousness/status' },
  { label: 'GET /consciousness/self-model', method: 'GET' as const, path: '/consciousness/self-model' },
  { label: 'POST /consciousness/reflect', method: 'POST' as const, path: '/consciousness/reflect', body: '{"text": "test input"}' },
  { label: 'GET /consciousness/health', method: 'GET' as const, path: '/consciousness/health' },
  { label: 'GET /consciousness/qualia', method: 'GET' as const, path: '/consciousness/qualia' },
  { label: 'POST /consciousness/feedback', method: 'POST' as const, path: '/consciousness/feedback', body: '{"episode_index": 0, "rating": 3}' },
  { label: 'POST /consciousness/seed?count=5', method: 'POST' as const, path: '/consciousness/seed?count=5' },
  { label: 'PATCH /consciousness/config', method: 'PATCH' as const, path: '/consciousness/config', body: '{"level": 2}' },
  { label: 'POST /consciousness/personality/presets/apply', method: 'POST' as const, path: '/consciousness/personality/presets/apply', body: '{"preset_id": "balanced"}' },
]

const METHOD_COLORS: Record<string, string> = {
  GET: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  POST: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  PATCH: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  DELETE: 'bg-red-500/15 text-red-400 border-red-500/30',
}

const SAMPLE_INPUTS = [
  'What is the meaning of consciousness?',
  'How do I learn new concepts effectively?',
  'Can you help me debug this code?',
  'I feel anxious about the future of AI',
  'Write a creative story about a sentient robot',
  'Explain the mathematical foundations of neural networks',
  'What are the ethical implications of artificial consciousness?',
  'Help me plan a productive work schedule',
]

export default function ConsciousnessTestingPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()

  const [selectedEndpoint, setSelectedEndpoint] = useState(0)
  const [customBody, setCustomBody] = useState('')
  const [apiResponse, setApiResponse] = useState<{ status: number; body: string; timeMs: number } | null>(null)
  const [apiLoading, setApiLoading] = useState(false)

  const [manualInput, setManualInput] = useState('')
  const [manualResult, setManualResult] = useState<Record<string, unknown> | null>(null)
  const [manualLoading, setManualLoading] = useState(false)

  const [batchCount, setBatchCount] = useState(5)
  const [batchResults, setBatchResults] = useState<Array<{ input: string; growth_delta: number; rating: number; qualia: Record<string, number>; narrative: string }>>([])
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchProgress, setBatchProgress] = useState(0)

  const [stateData, setStateData] = useState<Record<string, unknown> | null>(null)
  const [beliefsData, setBeliefsData] = useState<Record<string, unknown> | null>(null)
  const [episodesData, setEpisodesData] = useState<Array<Record<string, unknown>>>([])
  const [personalityData, setPersonalityData] = useState<Record<string, unknown> | null>(null)
  const [stateLoading, setStateLoading] = useState(false)

  const [resetLoading, setResetLoading] = useState(false)
  const [seedLoading, setSeedLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const ep = ENDPOINTS[selectedEndpoint]

  const handleSendRequest = useCallback(async () => {
    setApiLoading(true)
    setApiResponse(null)
    const start = Date.now()
    try {
      const init: RequestInit = {
        method: ep.method,
        headers: { 'Content-Type': 'application/json' },
      }
      const body = customBody || ep.body
      if (body && ep.method !== 'GET') {
        init.body = body
      }
      const res = await fetch(`${PUBLIC_API_URL}${ep.path}`, init)
      const elapsed = Date.now() - start
      const text = await res.text()
      let formatted = text
      try {
        formatted = JSON.stringify(JSON.parse(text), null, 2)
      } catch { /* raw text */ }
      setApiResponse({ status: res.status, body: formatted, timeMs: elapsed })
    } catch (e) {
      setApiResponse({ status: 0, body: extractErrorMessage(e), timeMs: Date.now() - start })
    } finally {
      setApiLoading(false)
    }
  }, [ep, customBody])

  const handleManualProcess = useCallback(async () => {
    if (!manualInput.trim()) {
      addToast('Please enter input text', 'error')
      return
    }
    setManualLoading(true)
    setManualResult(null)
    try {
      const data = await consciousnessController.reflect()
      setManualResult(data as Record<string, unknown>)
      addToast('Input processed', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setManualLoading(false)
    }
  }, [manualInput, addToast])

  const handleBatchRun = useCallback(async () => {
    setBatchLoading(true)
    setBatchResults([])
    setBatchProgress(0)
    const results: Array<{ input: string; growth_delta: number; rating: number; qualia: Record<string, number>; narrative: string }> = []

    for (let i = 0; i < batchCount; i++) {
      const input = SAMPLE_INPUTS[i % SAMPLE_INPUTS.length]
      try {
        const data = await consciousnessController.reflect()
        results.push({
          input,
          growth_delta: (data as any).growth_delta ?? 0,
          rating: (data as any).rating ?? 0,
          qualia: (data as any).qualia ?? {},
          narrative: (data as any).narrative ?? '',
        })
        setBatchResults([...results])
      } catch {
        results.push({
          input,
          growth_delta: 0,
          rating: 0,
          qualia: {},
          narrative: 'Error',
        })
        setBatchResults([...results])
      }
      setBatchProgress(i + 1)
    }
    setBatchLoading(false)
    addToast(`Batch complete: ${batchCount} iterations`, 'success')
  }, [batchCount, addToast])

  const loadStateInspector = useCallback(async () => {
    setStateLoading(true)
    try {
      const [statusData, selfModelData, episodesDataResult, personalityDataResult] = await Promise.all([
        consciousnessController.getStatus(),
        consciousnessController.getSelfModel(),
        consciousnessController.getEpisodeHistory(5),
        consciousnessController.getPersonality(),
      ])
      setStateData(statusData as unknown as Record<string, unknown>)
      setBeliefsData(selfModelData as unknown as Record<string, unknown>)
      setEpisodesData(((episodesDataResult as any)?.episodes || []).slice(-5))
      setPersonalityData(personalityDataResult as unknown as Record<string, unknown>)
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setStateLoading(false)
    }
  }, [addToast])

  const handleResetDefaults = useCallback(async () => {
    setResetLoading(true)
    try {
      await consciousnessController.resetPersonality()
      addToast('Reset to defaults', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setResetLoading(false)
    }
  }, [addToast])

  const handleSeed50 = useCallback(async () => {
    setSeedLoading(true)
    try {
      const data = await consciousnessController.seedData({ count: 50 })
      addToast(`Seeded ${(data as any).seeded ?? 50} episodes`, 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeedLoading(false)
    }
  }, [addToast])

  const handleExportState = useCallback(async () => {
    try {
      const data = await consciousnessController.getStatus()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-state-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('State exported', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [addToast])

  const handleImportState = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      JSON.parse(text)
      addToast('State imported successfully', 'success')
    } catch {
      addToast('Invalid JSON file', 'error')
    }
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [addToast])

  const batchAvgGrowth = batchResults.length > 0
    ? batchResults.reduce((sum, r) => sum + r.growth_delta, 0) / batchResults.length
    : 0
  const batchAvgRating = batchResults.length > 0
    ? batchResults.reduce((sum, r) => sum + r.rating, 0) / batchResults.length
    : 0

  return (
    <PageContainer title={t('consciousness_testing.page_title')}>
      <div className="space-y-6 p-6">

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_testing.api_tester_title')}</CardTitle>
            <CardDescription>{t('consciousness_testing.api_tester_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <select
                value={selectedEndpoint}
                onChange={(e) => {
                  setSelectedEndpoint(Number(e.target.value))
                  setCustomBody('')
                  setApiResponse(null)
                }}
                className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                {ENDPOINTS.map((ep, i) => (
                  <option key={i} value={i}>{ep.label}</option>
                ))}
              </select>
              <Button onClick={handleSendRequest} disabled={apiLoading}>
                {apiLoading ? t('consciousness_testing.sending') : t('consciousness_testing.send_request')}
              </Button>
            </div>
            {(ep.method === 'POST' || ep.method === 'PATCH') && (
              <textarea
                value={customBody}
                onChange={(e) => setCustomBody(e.target.value)}
                placeholder={ep.body || '{}'}
                className="w-full min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm font-mono resize-y focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            )}
            {apiResponse && (
              <div className="space-y-1">
                <div className="flex items-center gap-3 text-xs">
                  <span className={`font-mono font-medium ${
                    apiResponse.status >= 200 && apiResponse.status < 300 ? 'text-green-500' :
                    apiResponse.status >= 400 ? 'text-red-500' : 'text-muted-foreground'
                  }`}>
                    {apiResponse.status || 'ERR'}
                  </span>
                  <span className="text-muted-foreground">{apiResponse.timeMs}ms</span>
                </div>
                <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                  {apiResponse.body}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_testing.manual_title')}</CardTitle>
            <CardDescription>{t('consciousness_testing.manual_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <textarea
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder={t('consciousness_testing.manual_placeholder')}
              className="w-full min-h-[100px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <Button onClick={handleManualProcess} disabled={manualLoading || !manualInput.trim()}>
              {manualLoading ? t('consciousness_testing.processing') : t('consciousness_testing.process')}
            </Button>
            {manualResult && (
              <div className="space-y-2">
                {(manualResult as any).qualia && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.qualia')}</div>
                    <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                      {JSON.stringify((manualResult as any).qualia, null, 2)}
                    </pre>
                  </div>
                )}
                {(manualResult as any).beliefs && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.beliefs')}</div>
                    <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                      {JSON.stringify((manualResult as any).beliefs, null, 2)}
                    </pre>
                  </div>
                )}
                {(manualResult as any).narrative && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.narrative')}</div>
                    <div className="rounded-md border border-border/50 bg-muted/20 p-3 text-sm italic">
                      {String((manualResult as any).narrative)}
                    </div>
                  </div>
                )}
                {manualResult.growth_delta !== undefined && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">{t('consciousness_testing.growth_delta')}:</span>
                    <Badge variant={((manualResult.growth_delta as number) ?? 0) >= 0 ? 'default' : 'destructive'} className="text-xs">
                      {(((manualResult.growth_delta as number) ?? 0) >= 0 ? '+' : '')}{(((manualResult.growth_delta as number) ?? 0) * 100).toFixed(2)}%
                    </Badge>
                  </div>
                )}
                {manualResult.rating !== undefined && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">{t('consciousness_testing.rating')}:</span>
                    <Badge variant="outline" className="text-xs">{String(manualResult.rating)}</Badge>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_testing.batch_title')}</CardTitle>
            <CardDescription>{t('consciousness_testing.batch_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground">{t('consciousness_testing.iterations')}</label>
              <input
                type="number"
                min={1}
                max={50}
                value={batchCount}
                onChange={(e) => setBatchCount(Math.max(1, Math.min(50, Number(e.target.value))))}
                className="w-20 rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
              <Button onClick={handleBatchRun} disabled={batchLoading}>
                {batchLoading ? t('consciousness_testing.running_batch') : t('consciousness_testing.run_batch')}
              </Button>
              {batchLoading && (
                <span className="text-xs text-muted-foreground">{batchProgress}/{batchCount}</span>
              )}
            </div>
            {batchResults.length > 0 && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_testing.avg_growth')}</div>
                    <div className="font-mono">{(batchAvgGrowth * 100).toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">{t('consciousness_testing.avg_rating')}</div>
                    <div className="font-mono">{batchAvgRating.toFixed(1)}</div>
                  </div>
                </div>
                <div className="rounded-md border border-border/50 overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-muted/30">
                        <th className="px-3 py-1.5 text-left font-medium">#</th>
                        <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_testing.col_input')}</th>
                        <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_testing.col_growth')}</th>
                        <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_testing.col_rating')}</th>
                        <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_testing.col_narrative')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.map((r, i) => (
                        <tr key={i} className="border-t border-border/30">
                          <td className="px-3 py-1.5 font-mono">{i + 1}</td>
                          <td className="px-3 py-1.5 text-muted-foreground max-w-[200px] truncate">{r.input}</td>
                          <td className="px-3 py-1.5">
                            <Badge variant={r.growth_delta >= 0 ? 'default' : 'destructive'} className="text-[10px]">
                              {r.growth_delta >= 0 ? '+' : ''}{(r.growth_delta * 100).toFixed(1)}%
                            </Badge>
                          </td>
                          <td className="px-3 py-1.5">{r.rating}</td>
                          <td className="px-3 py-1.5 text-muted-foreground max-w-[200px] truncate italic">{r.narrative}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_testing.state_title')}</CardTitle>
            <CardDescription>{t('consciousness_testing.state_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button onClick={loadStateInspector} disabled={stateLoading} variant="outline">
              {stateLoading ? t('consciousness_testing.loading') : t('consciousness_testing.load_state')}
            </Button>
            {stateData && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.full_state')}</div>
                <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-48 whitespace-pre-wrap">
                  {JSON.stringify(stateData, null, 2)}
                </pre>
              </div>
            )}
            {beliefsData && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.beliefs_json')}</div>
                <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                  {JSON.stringify(beliefsData, null, 2)}
                </pre>
              </div>
            )}
            {episodesData.length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.episodes_json')}</div>
                <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-48 whitespace-pre-wrap">
                  {JSON.stringify(episodesData, null, 2)}
                </pre>
              </div>
            )}
            {personalityData && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_testing.personality_json')}</div>
                <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                  {JSON.stringify(personalityData, null, 2)}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_testing.quick_actions_title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={handleResetDefaults} disabled={resetLoading}>
                {resetLoading ? t('consciousness_testing.resetting') : t('consciousness_testing.reset_defaults')}
              </Button>
              <Button variant="outline" onClick={handleSeed50} disabled={seedLoading}>
                {seedLoading ? t('consciousness_testing.seeding') : t('consciousness_testing.seed_50')}
              </Button>
              <Button variant="outline" onClick={handleExportState}>
                {t('consciousness_testing.export_state')}
              </Button>
              <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                {t('consciousness_testing.import_state')}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleImportState}
                className="hidden"
              />
            </div>
          </CardContent>
        </Card>

      </div>
    </PageContainer>
  )
}
