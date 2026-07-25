'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button, Input } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { inferController, type InferRequest, type InferHealthResponse } from '@/lib/infer-controller'

type Tab = 'generate' | 'embed' | 'tokenize'

export default function InferPage() {
  const addToast = useToastStore(s => s.addToast)
  const [tab, setTab] = useState<Tab>('generate')
  const [health, setHealth] = useState<InferHealthResponse | null>(null)
  const [loadingHealth, setLoadingHealth] = useState(true)

  const fetchHealth = useCallback(async () => {
    try {
      const h = await inferController.health()
      setHealth(h)
    } catch { setHealth(null) }
    setLoadingHealth(false)
  }, [])

  useState(() => { fetchHealth() })

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Inference" />}
        right={
          loadingHealth ? <Spinner size="sm" /> : health ? (
            <Badge variant={health.model_loaded ? 'default' : 'secondary'} label={health.model_loaded ? health.model_id || 'Ready' : 'No model'} />
          ) : null
        }
      />
      <div className="space-y-4">
        <Tabs
          value={tab}
          onChange={v => setTab(v as Tab)}
          tabs={[
            { value: 'generate', label: 'Generate' },
            { value: 'embed', label: 'Embed' },
            { value: 'tokenize', label: 'Tokenize' },
          ]}
        />
        {tab === 'generate' && <GenerateTab />}
        {tab === 'embed' && <EmbedTab />}
        {tab === 'tokenize' && <TokenizeTab />}
      </div>
    </div>
  )
}

function GenerateTab() {
  const addToast = useToastStore(s => s.addToast)
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [maxTokens, setMaxTokens] = useState(128)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [meta, setMeta] = useState<{ tokens?: number; elapsed_ms?: number } | null>(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setLoading(true)
    setOutput('')
    setMeta(null)
    try {
      const res = await inferController.generate({ prompt, max_new_tokens: maxTokens, temperature, top_p: topP })
      setOutput(res.text)
      setMeta({ tokens: res.tokens_generated, elapsed_ms: res.elapsed_ms })
    } catch { addToast('Generation failed', 'error') }
    setLoading(false)
  }

  const handleStream = async () => {
    if (!prompt.trim()) return
    setStreaming(true)
    setOutput('')
    setMeta(null)
    try {
      let text = ''
      await inferController.generateStream(
        { prompt, max_new_tokens: maxTokens, temperature, top_p: topP },
        token => { text += token; setOutput(text) },
        m => { setMeta(m ?? null) },
        err => { addToast(err, 'error') },
      )
    } catch { addToast('Stream failed', 'error') }
    setStreaming(false)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Text Generation</CardTitle>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={handleStream} disabled={loading || streaming || !prompt.trim()}>
            {streaming ? <><Spinner size="sm" /> Streaming...</> : 'Stream'}
          </Button>
          <Button size="sm" onClick={handleGenerate} disabled={loading || streaming || !prompt.trim()}>
            {loading ? <><Spinner size="sm" /> Generating...</> : 'Generate'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <textarea
          className="w-full min-h-[80px] rounded-md border border-border bg-background px-3 py-2 text-sm resize-y focus:outline-none focus:ring-1 focus:ring-primary/30"
          placeholder="Enter a prompt..."
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleGenerate() }}
          aria-label="Prompt"
        />
        <div className="flex items-center gap-4 flex-wrap text-xs">
          <label className="flex items-center gap-1.5 text-muted-foreground">
            Max tokens
            <input type="number" className="w-16 h-7 rounded border border-border bg-background px-1.5 text-xs text-right" value={maxTokens} onChange={e => setMaxTokens(+e.target.value)} min={1} max={2048} />
          </label>
          <label className="flex items-center gap-1.5 text-muted-foreground">
            Temperature
            <input type="number" className="w-16 h-7 rounded border border-border bg-background px-1.5 text-xs text-right" value={temperature} onChange={e => setTemperature(+e.target.value)} min={0} max={2} step={0.1} />
          </label>
          <label className="flex items-center gap-1.5 text-muted-foreground">
            Top P
            <input type="number" className="w-16 h-7 rounded border border-border bg-background px-1.5 text-xs text-right" value={topP} onChange={e => setTopP(+e.target.value)} min={0} max={1} step={0.05} />
          </label>
        </div>
        {output && (
          <div className="rounded-md bg-zinc-950 p-4 font-mono text-sm text-zinc-100 leading-relaxed whitespace-pre-wrap min-h-[100px] max-h-[400px] overflow-auto">
            {output}
          </div>
        )}
        {meta && (
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {meta.tokens !== undefined && <span>{meta.tokens} tokens</span>}
            {meta.elapsed_ms !== undefined && <span>{(meta.elapsed_ms / 1000).toFixed(2)}s</span>}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function EmbedTab() {
  const addToast = useToastStore(s => s.addToast)
  const [text, setText] = useState('')
  const [result, setResult] = useState<{ embedding: number[]; dimensions: number; model: string } | null>(null)
  const [loading, setLoading] = useState(false)

  const handleEmbed = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await inferController.embed({ text })
      setResult(res)
    } catch { addToast('Embedding failed', 'error') }
    setLoading(false)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Text Embedding</CardTitle>
        <Button size="sm" onClick={handleEmbed} disabled={loading || !text.trim()}>
          {loading ? <><Spinner size="sm" /> Embedding...</> : 'Embed'}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Enter text to embed..."
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleEmbed() }}
          aria-label="Text to embed"
        />
        {result && (
          <div className="space-y-2">
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{result.dimensions}d</span>
              <span>{result.model}</span>
            </div>
            <div className="rounded-md bg-zinc-950 p-3 font-mono text-xs text-zinc-300 max-h-[200px] overflow-auto">
              [{result.embedding.slice(0, 20).map(v => v.toFixed(4)).join(', ')}{result.embedding.length > 20 ? ', ...' : ''}]
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function TokenizeTab() {
  const addToast = useToastStore(s => s.addToast)
  const [text, setText] = useState('')
  const [result, setResult] = useState<{ tokens: string[]; ids: number[]; count: number } | null>(null)
  const [loading, setLoading] = useState(false)

  const handleTokenize = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await inferController.tokenize({ text })
      setResult(res)
    } catch { addToast('Tokenization failed', 'error') }
    setLoading(false)
  }

  const handleDetokenize = async () => {
    if (!result?.ids.length) return
    setLoading(true)
    try {
      const res = await inferController.detokenize({ ids: result.ids })
      addToast(`Detokenized: "${res.text}"`, 'info')
    } catch { addToast('Detokenization failed', 'error') }
    setLoading(false)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Tokenization</CardTitle>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={handleDetokenize} disabled={loading || !result}>
            Detokenize
          </Button>
          <Button size="sm" onClick={handleTokenize} disabled={loading || !text.trim()}>
            {loading ? <><Spinner size="sm" /> Tokenizing...</> : 'Tokenize'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Enter text to tokenize..."
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleTokenize() }}
          aria-label="Text to tokenize"
        />
        {result && (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">{result.count} tokens</div>
            <div className="flex flex-wrap gap-1.5">
              {result.tokens.map((tok, i) => (
                <span key={i} className="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-mono text-primary">
                  {tok}
                  <span className="text-muted-foreground text-[10px]">{result.ids[i]}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
