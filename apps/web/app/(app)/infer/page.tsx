'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Textarea } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { useToastStore } from '@/lib/toast-store'
import {
  inferController,
  type InferResponse,
  type EmbedResponse,
  type TokenizeResponse,
  type InferHealth,
  type InferInfo,
} from '@/lib/infer-controller'

type Tab = 'generate' | 'embed' | 'tokenize' | 'info'

export default function InferPage() {
  const addToast = useToastStore(s => s.addToast)
  const [tab, setTab] = useState<Tab>('generate')
  const [health, setHealth] = useState<InferHealth | null>(null)
  const [loading, setLoading] = useState(false)

  const [prompt, setPrompt] = useState('Hello, how are you?')
  const [maxTokens, setMaxTokens] = useState(256)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.85)
  const [topK, setTopK] = useState(40)
  const [repPenalty, setRepPenalty] = useState(1.15)

  const [genResult, setGenResult] = useState<InferResponse | null>(null)
  const [embedResult, setEmbedResult] = useState<EmbedResponse | null>(null)
  const [tokenResult, setTokenResult] = useState<TokenizeResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<InferInfo | null>(null)

  const checkHealth = useCallback(async () => {
    try {
      const h = await inferController.health()
      setHealth(h)
    } catch {
      setHealth({ status: 'unreachable', model_loaded: false, has_streaming: false, has_embedding: false })
    }
  }, [])

  useEffect(() => { void checkHealth() }, [checkHealth])

  const handleGenerate = async () => {
    setLoading(true)
    setGenResult(null)
    try {
      const result = await inferController.generate({
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repPenalty,
      })
      setGenResult(result)
    } catch (e) {
      addToast(`Generation failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleEmbed = async () => {
    setLoading(true)
    setEmbedResult(null)
    try {
      const result = await inferController.embed(prompt)
      setEmbedResult(result)
    } catch (e) {
      addToast(`Embed failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleTokenize = async () => {
    setLoading(true)
    setTokenResult(null)
    try {
      const result = await inferController.tokenize(prompt)
      setTokenResult(result)
    } catch (e) {
      addToast(`Tokenize failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadInfo = async () => {
    setLoading(true)
    setModelInfo(null)
    try {
      const result = await inferController.info()
      setModelInfo(result)
    } catch (e) {
      addToast(`Could not load model info: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const tabs: { key: Tab; label: string; available: boolean }[] = [
    { key: 'generate', label: 'Generate', available: true },
    { key: 'embed', label: 'Embed', available: health?.has_embedding ?? false },
    { key: 'tokenize', label: 'Tokenize', available: true },
    { key: 'info', label: 'Model Info', available: true },
  ]

  return (
    <PageContainer
      title="API Playground"
      subtitle="Test inference endpoints directly"
      headerRight={
        <div className="flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${health?.model_loaded ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-muted-foreground">{health?.status ?? 'checking...'}</span>
        </div>
      }
    >
      <div className="space-y-4">
        <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-1">
          {tabs.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              disabled={!t.available}
              className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                tab === t.key ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground disabled:opacity-40'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {tab === 'generate' && 'Text Generation'}
              {tab === 'embed' && 'Text Embedding'}
              {tab === 'tokenize' && 'Tokenization'}
              {tab === 'info' && 'Model Information'}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {tab !== 'info' && (
              <div className="space-y-2">
                <Label className="text-xs">Input</Label>
                <Textarea
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  rows={4}
                  className="font-mono text-xs"
                  placeholder="Enter text..."
                />
              </div>
            )}

            {tab === 'generate' && (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                <div className="space-y-1">
                  <Label className="text-xs">Max tokens</Label>
                  <Input type="number" value={maxTokens} onChange={e => setMaxTokens(Number(e.target.value))} className="h-8 text-xs" />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Temperature</Label>
                  <Input type="number" step="0.05" value={temperature} onChange={e => setTemperature(Number(e.target.value))} className="h-8 text-xs" />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Top P</Label>
                  <Input type="number" step="0.05" value={topP} onChange={e => setTopP(Number(e.target.value))} className="h-8 text-xs" />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Top K</Label>
                  <Input type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} className="h-8 text-xs" />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Rep Penalty</Label>
                  <Input type="number" step="0.05" value={repPenalty} onChange={e => setRepPenalty(Number(e.target.value))} className="h-8 text-xs" />
                </div>
              </div>
            )}

            <Button
              onClick={() => {
                if (tab === 'generate') void handleGenerate()
                else if (tab === 'embed') void handleEmbed()
                else if (tab === 'tokenize') void handleTokenize()
                else void handleLoadInfo()
              }}
              disabled={loading || !health?.model_loaded}
              className="w-full"
            >
              {loading ? 'Running...' : tab === 'info' ? 'Load Info' : `Run ${tab}`}
            </Button>
          </CardContent>
        </Card>

        {genResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <pre className="max-h-[300px] overflow-y-auto rounded bg-muted/30 p-3 text-xs whitespace-pre-wrap">{genResult.text}</pre>
              <div className="flex gap-4 text-xs text-muted-foreground">
                <span>{genResult.tokens_generated} tokens</span>
                <span>{genResult.elapsed_ms.toFixed(0)}ms</span>
                <span>{genResult.model}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {embedResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Embedding</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-4 text-xs text-muted-foreground">
                <span>{embedResult.dimensions} dimensions</span>
                <span>{embedResult.model}</span>
              </div>
              <div className="flex gap-1">
                {embedResult.embedding.slice(0, 20).map((v, i) => (
                  <div key={i} className="h-8 w-3 rounded-t" style={{ backgroundColor: `rgb(${Math.round(Math.abs(v) * 255)}, ${Math.round(Math.abs(v) * 100)}, ${Math.round((1 - Math.abs(v)) * 200)})` }} title={`${i}: ${v.toFixed(4)}`} />
                ))}
                {embedResult.dimensions > 20 && <span className="text-xs text-muted-foreground self-end">+{embedResult.dimensions - 20}</span>}
              </div>
              <pre className="max-h-[150px] overflow-y-auto rounded bg-muted/30 p-3 text-xs">{JSON.stringify(embedResult.embedding.slice(0, 10), null, 2)}{embedResult.dimensions > 10 ? '\n...' : ''}</pre>
            </CardContent>
          </Card>
        )}

        {tokenResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tokens ({tokenResult.count})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-1">
                {tokenResult.tokens.map((t, i) => (
                  <span key={i} className="inline-flex items-center gap-1 rounded border border-border bg-muted/30 px-1.5 py-0.5 text-[10px]">
                    <span className="font-mono text-muted-foreground">{tokenResult.ids[i]}</span>
                    <span>{t}</span>
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {modelInfo && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Model Info</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {[
                  { label: 'Model ID', value: modelInfo.model_id },
                  { label: 'Type', value: modelInfo.model_type },
                  { label: 'Parameters', value: modelInfo.num_parameters.toLocaleString() },
                  { label: 'Vocab Size', value: modelInfo.vocab_size.toLocaleString() },
                  { label: 'Max Context', value: modelInfo.max_context.toLocaleString() },
                  { label: 'Layers', value: modelInfo.num_layers },
                  { label: 'Tokenizer', value: modelInfo.has_tokenizer ? 'Yes' : 'No' },
                  { label: 'Streaming', value: modelInfo.has_streaming ? 'Yes' : 'No' },
                  { label: 'Embedding', value: modelInfo.has_embedding ? 'Yes' : 'No' },
                ].map(item => (
                  <div key={item.label} className="rounded bg-muted/30 p-2">
                    <div className="text-[10px] text-muted-foreground">{item.label}</div>
                    <div className="text-xs font-medium">{String(item.value)}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
