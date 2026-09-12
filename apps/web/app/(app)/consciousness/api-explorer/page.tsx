'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
  Input,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { IconCode, IconPlus, IconTrash, IconClock } from '@/components/icons/NavIcons'

type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'

interface Param {
  name: string
  type: string
  required: boolean
  description: string
}

interface Endpoint {
  method: HttpMethod
  path: string
  description: string
  params?: Param[]
  requestBody?: string
}

type Category = 'core' | 'history' | 'training' | 'personality' | 'personas' | 'data' | 'advanced'

const CATEGORIES: { id: Category; labelKey: string }[] = [
  { id: 'core', labelKey: 'consciousness_api_explorer.cat_core' },
  { id: 'history', labelKey: 'consciousness_api_explorer.cat_history' },
  { id: 'training', labelKey: 'consciousness_api_explorer.cat_training' },
  { id: 'personality', labelKey: 'consciousness_api_explorer.cat_personality' },
  { id: 'personas', labelKey: 'consciousness_api_explorer.cat_personas' },
  { id: 'data', labelKey: 'consciousness_api_explorer.cat_data' },
  { id: 'advanced', labelKey: 'consciousness_api_explorer.cat_advanced' },
]

const ENDPOINTS: Record<Category, Endpoint[]> = {
  core: [
    { method: 'GET', path: '/consciousness/status', description: 'Get consciousness status' },
    { method: 'GET', path: '/consciousness/self-model', description: 'Get self-model data' },
    { method: 'GET', path: '/consciousness/qualia', description: 'Get current qualia state' },
    { method: 'POST', path: '/consciousness/reflect', description: 'Trigger self-reflection', requestBody: '{\n  "text": "Your reflection prompt here"\n}' },
    { method: 'PATCH', path: '/consciousness/config', description: 'Update consciousness level', requestBody: '{\n  "level": 2\n}', params: [{ name: 'level', type: 'number', required: true, description: 'Consciousness level (0-3)' }] },
    { method: 'GET', path: '/consciousness/health', description: 'Health check' },
  ],
  history: [
    { method: 'GET', path: '/consciousness/history/episodes', description: 'Get episode history', params: [{ name: 'limit', type: 'number', required: false, description: 'Max episodes (default 20)' }, { name: 'offset', type: 'number', required: false, description: 'Skip count' }] },
    { method: 'GET', path: '/consciousness/history/qualia', description: 'Get qualia history', params: [{ name: 'limit', type: 'number', required: false, description: 'Max entries (default 50)' }] },
    { method: 'GET', path: '/consciousness/history/beliefs', description: 'Get beliefs evolution' },
    { method: 'POST', path: '/consciousness/feedback', description: 'Rate an episode', requestBody: '{\n  "episode_index": 0,\n  "rating": 5\n}', params: [{ name: 'episode_index', type: 'number', required: true, description: 'Episode index' }, { name: 'rating', type: 'number', required: true, description: 'Rating 1-5' }] },
  ],
  training: [
    { method: 'GET', path: '/consciousness/train/status', description: 'Get training status' },
    { method: 'POST', path: '/consciousness/train/start', description: 'Start training', requestBody: '{\n  "model_path": "./models/base"\n}' },
    { method: 'GET', path: '/consciousness/evaluate', description: 'Evaluate system quality' },
  ],
  personality: [
    { method: 'GET', path: '/consciousness/personality', description: 'Get personality profile' },
    { method: 'PATCH', path: '/consciousness/personality', description: 'Update personality', requestBody: '{\n  "traits": {\n    "openness": 0.8\n  }\n}' },
    { method: 'POST', path: '/consciousness/personality/reset', description: 'Reset to defaults' },
    { method: 'GET', path: '/consciousness/personality/history', description: 'Get personality history' },
    { method: 'GET', path: '/consciousness/personality/presets', description: 'List presets' },
    { method: 'POST', path: '/consciousness/personality/presets/apply', description: 'Apply preset', requestBody: '{\n  "preset_id": "empathetic"\n}', params: [{ name: 'preset_id', type: 'string', required: true, description: 'Preset identifier' }] },
    { method: 'GET', path: '/consciousness/personality/conflicts', description: 'Detect conflicts' },
  ],
  personas: [
    { method: 'GET', path: '/consciousness/personas', description: 'List saved personas' },
    { method: 'POST', path: '/consciousness/personas/save', description: 'Save a persona', requestBody: '{\n  "name": "My Persona",\n  "traits": {\n    "openness": 0.8\n  }\n}' },
    { method: 'GET', path: '/consciousness/personas/:id', description: 'Get a persona', params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }] },
    { method: 'POST', path: '/consciousness/personas/:id/activate', description: 'Activate a persona', params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }] },
    { method: 'DELETE', path: '/consciousness/personas/:id', description: 'Delete a persona', params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }] },
  ],
  data: [
    { method: 'POST', path: '/consciousness/seed', description: 'Seed test data', params: [{ name: 'count', type: 'number', required: false, description: 'Number of episodes (default 10)' }] },
  ],
  advanced: [
    { method: 'POST', path: '/consciousness/process', description: 'Process input text', requestBody: '{\n  "text": "Hello world"\n}' },
    { method: 'POST', path: '/consciousness/batch', description: 'Batch process inputs', requestBody: '{\n  "inputs": ["Hello", "World"]\n}' },
    { method: 'POST', path: '/consciousness/reset', description: 'Reset consciousness state' },
    { method: 'POST', path: '/consciousness/backup', description: 'Create backup snapshot' },
    { method: 'GET', path: '/consciousness/backup', description: 'List backups' },
    { method: 'GET', path: '/consciousness/backup/download', description: 'Download backup as file' },
  ],
}

const METHOD_COLORS: Record<HttpMethod, string> = {
  GET: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  POST: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  PUT: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  PATCH: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  DELETE: 'bg-red-500/15 text-red-400 border-red-500/30',
}

const ALL_ENDPOINTS = Object.values(ENDPOINTS).flat()

interface SavedRequest {
  id: string
  name: string
  method: HttpMethod
  path: string
  body: string
  params: { key: string; value: string }[]
  headers: { key: string; value: string }[]
  tag: string
  timestamp: number
}

interface HistoryEntry {
  id: string
  method: HttpMethod
  path: string
  status: number
  timeMs: number
  timestamp: number
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

export default function ConsciousnessApiExplorerPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()

  const [selectedIdx, setSelectedIdx] = useState(0)
  const [params, setParams] = useState<{ key: string; value: string }[]>([])
  const [body, setBody] = useState('')
  const [headers, setHeaders] = useState<{ key: string; value: string }[]>([{ key: 'Content-Type', value: 'application/json' }])
  const [response, setResponse] = useState<{ status: number; body: string; timeMs: number; headers: Record<string, string> } | null>(null)
  const [loading, setLoading] = useState(false)
  const [collection, setCollection] = useState<SavedRequest[]>([])
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [saveName, setSaveName] = useState('')
  const [saveTag, setSaveTag] = useState('')
  const [showSave, setShowSave] = useState(false)
  const responseRef = useRef<HTMLDivElement>(null)

  const endpoint = ALL_ENDPOINTS[selectedIdx] as Endpoint

  useEffect(() => {
    try {
      const saved = localStorage.getItem('consciousness_api_collection')
      if (saved) setCollection(JSON.parse(saved))
      const hist = localStorage.getItem('consciousness_api_history')
      if (hist) setHistory(JSON.parse(hist))
    } catch { /* ignore */ }
  }, [])

  const prevIdxRef = useRef(selectedIdx)
  useEffect(() => {
    if (prevIdxRef.current !== selectedIdx) {
      prevIdxRef.current = selectedIdx
      setParams(endpoint.params?.map(() => ({ key: '', value: '' })) ?? [])
      setBody(endpoint.requestBody ?? '')
      setResponse(null)
    }
  }, [selectedIdx, endpoint.params, endpoint.requestBody])

  const buildUrl = useCallback((currentParams: { key: string; value: string }[]) => {
    let path = endpoint.path
    endpoint.params?.forEach((p, i) => {
      if (currentParams[i]?.value) {
        path = path.replace(`:${p.name}`, currentParams[i].value)
      }
    })
    const qs = endpoint.params
      ?.filter((p, i) => !path.includes(`:${p.name}`) && currentParams[i]?.value)
      .map((p, i) => `${p.name}=${encodeURIComponent(currentParams[i].value)}`)
      .join('&')
    return `${PUBLIC_API_URL}${path}${qs ? `?${qs}` : ''}`
  }, [endpoint])

  const handleSend = useCallback(async () => {
    setLoading(true)
    setResponse(null)
    const start = Date.now()
    try {
      const init: RequestInit = {
        method: endpoint.method,
        headers: {} as Record<string, string>,
      }
      headers.forEach(h => {
        if (h.key && h.value) (init.headers as Record<string, string>)[h.key] = h.value
      })
      if (body && endpoint.method !== 'GET') {
        init.body = body
      }
      const res = await fetch(buildUrl(params), init)
      const elapsed = Date.now() - start
      const text = await res.text()
      const resHeaders: Record<string, string> = {}
      res.headers.forEach((v, k) => { resHeaders[k] = v })
      let formatted = text
      try { formatted = JSON.stringify(JSON.parse(text), null, 2) } catch { /* raw */ }
      setResponse({ status: res.status, body: formatted, timeMs: elapsed, headers: resHeaders })

      const entry: HistoryEntry = {
        id: generateId(),
        method: endpoint.method,
        path: endpoint.path,
        status: res.status,
        timeMs: elapsed,
        timestamp: Date.now(),
      }
      setHistory(prev => {
        const next = [entry, ...prev].slice(0, 20)
        try { localStorage.setItem('consciousness_api_history', JSON.stringify(next)) } catch { /* ignore */ }
        return next
      })
    } catch (e) {
      const elapsed = Date.now() - start
      setResponse({ status: 0, body: extractErrorMessage(e), timeMs: elapsed, headers: {} })
      const entry: HistoryEntry = {
        id: generateId(),
        method: endpoint.method,
        path: endpoint.path,
        status: 0,
        timeMs: elapsed,
        timestamp: Date.now(),
      }
      setHistory(prev => {
        const next = [entry, ...prev].slice(0, 20)
        try { localStorage.setItem('consciousness_api_history', JSON.stringify(next)) } catch { /* ignore */ }
        return next
      })
    } finally {
      setLoading(false)
      setTimeout(() => { responseRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }, 50)
    }
  }, [endpoint, params, body, headers, buildUrl])

  const handleSave = useCallback(() => {
    if (!saveName.trim()) return
    const req: SavedRequest = {
      id: generateId(),
      name: saveName.trim(),
      method: endpoint.method,
      path: endpoint.path,
      body,
      params,
      headers,
      tag: saveTag.trim(),
      timestamp: Date.now(),
    }
    const next = [req, ...collection]
    setCollection(next)
    try { localStorage.setItem('consciousness_api_collection', JSON.stringify(next)) } catch { /* ignore */ }
    setSaveName('')
    setSaveTag('')
    setShowSave(false)
    addToast(t('consciousness_api_explorer.toast_saved'), 'success')
  }, [saveName, saveTag, endpoint, body, params, headers, collection, addToast, t])

  const handleLoad = useCallback((req: SavedRequest) => {
    const idx = ALL_ENDPOINTS.findIndex(e => e.method === req.method && e.path === req.path)
    if (idx >= 0) {
      setSelectedIdx(idx)
      setBody(req.body)
      setParams(req.params.length > 0 ? req.params : endpoint.params?.map(() => ({ key: '', value: '' })) ?? [])
      setHeaders(req.headers.length > 0 ? req.headers : [{ key: 'Content-Type', value: 'application/json' }])
    }
  }, [endpoint])

  const handleDeleteCollection = useCallback((id: string) => {
    const next = collection.filter(r => r.id !== id)
    setCollection(next)
    try { localStorage.setItem('consciousness_api_collection', JSON.stringify(next)) } catch { /* ignore */ }
  }, [collection])

  const handleReplay = useCallback((entry: HistoryEntry) => {
    const idx = ALL_ENDPOINTS.findIndex(e => e.method === entry.method && e.path === entry.path)
    if (idx >= 0) setSelectedIdx(idx)
  }, [])

  const handleDeleteHistory = useCallback((id: string) => {
    const next = history.filter(h => h.id !== id)
    setHistory(next)
    try { localStorage.setItem('consciousness_api_history', JSON.stringify(next)) } catch { /* ignore */ }
  }, [history])

  return (
    <PageContainer title={t('consciousness_api_explorer.page_title')}>
      <div className="flex gap-4 min-h-[calc(100vh-6rem)]">
        <div className="w-72 shrink-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-xs">{t('consciousness_api_explorer.endpoints')}</CardTitle>
            </CardHeader>
            <CardContent className="p-2">
              <div className="max-h-[50vh] overflow-y-auto space-y-1">
                {CATEGORIES.map(cat => (
                  <div key={cat.id}>
                    <p className="px-2 py-1 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                      {t(cat.labelKey)}
                    </p>
                    {ENDPOINTS[cat.id].map(ep => {
                      const globalIdx = ALL_ENDPOINTS.indexOf(ep)
                      return (
                        <button
                          key={`${ep.method}:${ep.path}`}
                          type="button"
                          onClick={() => setSelectedIdx(globalIdx)}
                          className={`w-full text-left px-2 py-1.5 rounded text-[11px] flex items-center gap-1.5 transition-colors ${
                            globalIdx === selectedIdx
                              ? 'bg-primary/10 text-primary'
                              : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                          }`}
                        >
                          <Badge variant="outline" className={`font-mono text-[9px] border px-1 py-0 shrink-0 ${METHOD_COLORS[ep.method]}`}>
                            {ep.method}
                          </Badge>
                          <span className="truncate font-mono">{ep.path}</span>
                        </button>
                      )
                    })}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs flex items-center gap-1.5">
                  <IconClock className="h-3 w-3" />
                  {t('consciousness_api_explorer.history')}
                </CardTitle>
                <span className="text-[10px] text-muted-foreground">{history.length}</span>
              </div>
            </CardHeader>
            <CardContent className="p-2">
              <div className="max-h-48 overflow-y-auto space-y-1">
                {history.length === 0 && (
                  <p className="text-[10px] text-muted-foreground text-center py-2">{t('consciousness_api_explorer.no_history')}</p>
                )}
                {history.map(h => (
                  <div
                    key={h.id}
                    className="flex items-center gap-1.5 px-2 py-1 rounded hover:bg-muted/50 text-[10px] cursor-pointer group"
                    onClick={() => handleReplay(h)}
                  >
                    <Badge variant="outline" className={`font-mono text-[8px] border px-1 py-0 shrink-0 ${METHOD_COLORS[h.method]}`}>
                      {h.method}
                    </Badge>
                    <span className={`font-mono ${
                      h.status >= 200 && h.status < 300 ? 'text-green-500' :
                      h.status >= 400 ? 'text-red-500' : 'text-muted-foreground'
                    }`}>
                      {h.status || 'ERR'}
                    </span>
                    <span className="text-muted-foreground">{h.timeMs}ms</span>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); handleDeleteHistory(h.id) }}
                      className="ml-auto opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground"
                    >
                      <IconTrash className="h-2.5 w-2.5" />
                    </button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex-1 space-y-4 min-w-0">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={`font-mono text-xs border ${METHOD_COLORS[endpoint.method]}`}>
                    {endpoint.method}
                  </Badge>
                  <code className="text-sm font-mono">{endpoint.path}</code>
                </div>
                <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setShowSave(!showSave)}>
                  <IconPlus className="h-3 w-3 mr-1" />
                  {t('consciousness_api_explorer.save')}
                </Button>
              </div>
              <CardDescription>{endpoint.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {showSave && (
                <div className="flex items-center gap-2 p-2 rounded border border-border/50 bg-muted/20">
                  <Input
                    value={saveName}
                    onChange={(e) => setSaveName((e.target as HTMLInputElement).value)}
                    placeholder={t('consciousness_api_explorer.save_name')}
                    className="h-7 text-xs"
                  />
                  <Input
                    value={saveTag}
                    onChange={(e) => setSaveTag((e.target as HTMLInputElement).value)}
                    placeholder={t('consciousness_api_explorer.save_tag')}
                    className="h-7 text-xs w-24"
                  />
                  <Button size="sm" className="h-7 text-xs" onClick={handleSave}>
                    {t('consciousness_api_explorer.save_confirm')}
                  </Button>
                </div>
              )}

              {endpoint.params && endpoint.params.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground mb-2">{t('consciousness_api_explorer.parameters')}</h4>
                  <div className="space-y-1.5">
                    {endpoint.params.map((p, i) => (
                      <div key={p.name} className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-muted-foreground w-24 shrink-0">
                          {p.name}
                          {p.required && <span className="text-red-400">*</span>}
                        </span>
                        <Input
                          value={params[i]?.value ?? ''}
                          onChange={(e) => {
                            const next = [...params]
                            next[i] = { ...next[i], value: (e.target as HTMLInputElement).value }
                            setParams(next)
                          }}
                          placeholder={p.description}
                          className="h-7 text-xs flex-1"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(endpoint.method === 'POST' || endpoint.method === 'PATCH' || endpoint.method === 'PUT') && (
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground mb-2">{t('consciousness_api_explorer.body')}</h4>
                  <textarea
                    value={body}
                    onChange={(e) => setBody((e.target as HTMLTextAreaElement).value)}
                    className="w-full h-32 rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono resize-y focus:outline-none focus:ring-1 focus:ring-primary/50"
                    placeholder="{}"
                  />
                </div>
              )}

              <div>
                <h4 className="text-xs font-medium text-muted-foreground mb-2">{t('consciousness_api_explorer.headers')}</h4>
                <div className="space-y-1.5">
                  {headers.map((h, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Input
                        value={h.key}
                        onChange={(e) => {
                          const next = [...headers]
                          next[i] = { ...next[i], key: (e.target as HTMLInputElement).value }
                          setHeaders(next)
                        }}
                        placeholder="Key"
                        className="h-7 text-xs flex-1"
                      />
                      <Input
                        value={h.value}
                        onChange={(e) => {
                          const next = [...headers]
                          next[i] = { ...next[i], value: (e.target as HTMLInputElement).value }
                          setHeaders(next)
                        }}
                        placeholder="Value"
                        className="h-7 text-xs flex-1"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 w-7 p-0 shrink-0"
                        onClick={() => setHeaders(headers.filter((_, j) => j !== i))}
                      >
                        <IconTrash className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setHeaders([...headers, { key: '', value: '' }])}
                  >
                    <IconPlus className="h-3 w-3 mr-1" />
                    {t('consciousness_api_explorer.add_header')}
                  </Button>
                </div>
              </div>

              <Button onClick={handleSend} disabled={loading} className="w-full">
                {loading ? t('consciousness_api_explorer.sending') : t('consciousness_api_explorer.send_request')}
              </Button>
            </CardContent>
          </Card>

          {response && (
            <div ref={responseRef}>
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className={`font-mono text-xs border ${
                      response.status >= 200 && response.status < 300
                        ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                        : response.status >= 400
                          ? 'bg-red-500/15 text-red-400 border-red-500/30'
                          : 'bg-muted/50 text-muted-foreground border-border/50'
                    }`}>
                      {response.status || 'ERR'}
                    </Badge>
                    <span className="text-xs text-muted-foreground">{response.timeMs}ms</span>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.keys(response.headers).length > 0 && (
                    <div>
                      <h4 className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_api_explorer.response_headers')}</h4>
                      <div className="rounded-md border border-border/50 bg-muted/20 p-2 space-y-0.5">
                        {Object.entries(response.headers).map(([k, v]) => (
                          <div key={k} className="text-[10px] font-mono">
                            <span className="text-primary">{k}</span>
                            <span className="text-muted-foreground">: {v}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div>
                    <h4 className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_api_explorer.response_body')}</h4>
                    <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-80 whitespace-pre-wrap">
                      {response.body}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {collection.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-xs">{t('consciousness_api_explorer.collection')}</CardTitle>
              </CardHeader>
              <CardContent className="p-2">
                <div className="space-y-1">
                  {collection.map(req => (
                    <div
                      key={req.id}
                      className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted/50 text-[11px] cursor-pointer group"
                      onClick={() => handleLoad(req)}
                    >
                      <Badge variant="outline" className={`font-mono text-[9px] border px-1 py-0 shrink-0 ${METHOD_COLORS[req.method]}`}>
                        {req.method}
                      </Badge>
                      <span className="font-medium truncate">{req.name}</span>
                      {req.tag && (
                        <Badge variant="secondary" className="text-[8px] px-1 py-0 shrink-0">{req.tag}</Badge>
                      )}
                      <span className="text-muted-foreground truncate font-mono text-[9px]">{req.path}</span>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleDeleteCollection(req.id) }}
                        className="ml-auto opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground"
                      >
                        <IconTrash className="h-2.5 w-2.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </PageContainer>
  )
}
