'use client'

import { useState, useRef, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import {
  SoulNetWebGPU,
  SoulTransformerWebGPU,
  inferArch,
  parseSou,
} from '@/lib/soulnet-webgpu'
import { soulsController } from '@/lib/souls-controller'
import { cn } from '@/lib/cn'

type EngineState = 'idle' | 'loading' | 'ready' | 'generating' | 'error'

export default function WebGPUPlaygroundPage() {
  const [engineState, setEngineState] = useState<EngineState>('idle')
  const [archInfo, setArchInfo] = useState<string | null>(null)
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null)
  const [modelUrl, setModelUrl] = useState('')
  const [prompt, setPrompt] = useState('hello')
  const [output, setOutput] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [error, setError] = useState<string | null>(null)
  const [webgpuAvailable, setWebgpuAvailable] = useState<boolean | null>(null)
  const [checkpoints, setCheckpoints] = useState<{ name: string; path: string }[]>([])

  const engineRef = useRef<SoulNetWebGPU | SoulTransformerWebGPU | null>(null)
  const abortRef = useRef(false)

  useState(() => {
    setWebgpuAvailable(!!navigator.gpu)
    soulsController.listCheckpoints().then((list) => {
      if (list && Array.isArray(list)) {
        setCheckpoints(list.map((c: any) => ({ name: c.name, path: `/auto-train/checkpoints/${c.name}/download` })))
      }
    }).catch(() => {})
  })

  const loadEngine = useCallback(async (url: string) => {
    if (!url) return
    setEngineState('loading')
    setError(null)
    setOutput('')
    try {
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const buf = await resp.arrayBuffer()
      const arch = inferArch(buf)
      const cp = parseSou(buf)
      setMetadata(cp.metadata as any)

      if (arch.archType === 'transformer') {
        const engine = new SoulTransformerWebGPU()
        await engine.init()
        await engine.load(buf, {
          archType: 'transformer',
          embedDim: arch.embedDim,
          numHeads: 8,
          numKVHeads: 8,
          numLayers: arch.numLayers,
          dimFF: 1024,
          vocabSize: arch.vocabSize,
          maxSeqLen: 2048,
          eps: 1e-5,
        })
        engineRef.current = engine
        setArchInfo(`Transformer ${arch.embedDim}×${arch.numLayers}L`)
      } else {
        const engine = new SoulNetWebGPU()
        await engine.init()
        await engine.load(buf, arch)
        engineRef.current = engine
        setArchInfo(`LSTM ${arch.embedDim}×${arch.hiddenDim}`)
      }
      setEngineState('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load failed')
      setEngineState('error')
    }
  }, [])

  const generate = useCallback(async () => {
    if (!engineRef.current || engineState === 'generating') return
    setEngineState('generating')
    setOutput('')
    abortRef.current = false
    try {
      const gen = engineRef.current.generate(prompt, maxTokens, temperature)
      let text = ''
      for await (const token of gen) {
        if (abortRef.current) break
        text += token
        setOutput(text)
      }
      setEngineState('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed')
      setEngineState('error')
    }
  }, [prompt, temperature, maxTokens, engineState])

  const stop = useCallback(() => {
    abortRef.current = true
    setEngineState('ready')
  }, [])

  const reset = useCallback(() => {
    engineRef.current?.resetState?.()
    setOutput('')
  }, [])

  const destroy = useCallback(() => {
    engineRef.current?.destroy?.()
    engineRef.current = null
    setEngineState('idle')
    setArchInfo(null)
    setMetadata(null)
    setOutput('')
  }, [])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="WebGPU Playground" />} />
      <div className="space-y-4">
        {!webgpuAvailable && (
          <Card className="border-warning">
            <CardContent className="py-3 text-sm text-warning">
              WebGPU is not available in this browser. Chrome 113+ or Edge 113+ required.
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Model</CardTitle>
              {engineState === 'ready' && archInfo && (
                <Badge variant="success">{archInfo}</Badge>
              )}
              {engineState === 'loading' && <Badge variant="warning">Loading...</Badge>}
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input
                value={modelUrl}
                onChange={(e) => setModelUrl(e.target.value)}
                placeholder="/auto-train/checkpoints/friendly.sou"
                className="text-sm"
                aria-label="Model file path"
              />
              <Button
                size="sm"
                onClick={() => loadEngine(modelUrl)}
                disabled={!modelUrl || engineState === 'loading' || engineState === 'generating'}
              >
                {engineState === 'loading' ? 'Loading...' : 'Load'}
              </Button>
              {engineState === 'ready' && (
                <Button size="sm" variant="destructive" onClick={destroy}>
                  Unload
                </Button>
              )}
            </div>

            {checkpoints.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {checkpoints.map((cp) => (
                  <Button
                    key={cp.name}
                    size="sm"
                    variant="ghost"
                    className="h-7 text-xs"
                    onClick={() => {
                      setModelUrl(cp.path)
                      loadEngine(cp.path)
                    }}
                  >
                    {cp.name}
                  </Button>
                ))}
              </div>
            )}

            {metadata && (
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                {metadata.soul_name != null && <div>Soul: {String(metadata.soul_name)}</div>}
                {metadata.lineage != null && <div>Lineage: {String(metadata.lineage)}</div>}
                {metadata.step != null && <div>Step: {String(metadata.step)}</div>}
                {metadata.system_prompt != null && (
                  <div className="col-span-2 mt-1 truncate">
                    Prompt: {String(metadata.system_prompt).slice(0, 120)}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Generate</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter prompt..."
              rows={3}
              className="text-sm font-mono"
            />

            <div className="flex items-center gap-4 text-sm">
              <label className="flex items-center gap-2">
                <span className="text-muted-foreground">Temp:</span>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-20"
                />
                <span className="w-8 text-xs">{temperature.toFixed(1)}</span>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-muted-foreground">Max:</span>
                <Input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 200)}
                  className="w-16 h-7 text-xs"
                />
              </label>
            </div>

            <div className="flex gap-2">
              {engineState === 'generating' ? (
                <Button size="sm" variant="destructive" onClick={stop}>
                  Stop
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={generate}
                  disabled={engineState !== 'ready' || !prompt}
                >
                  Generate
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={reset} disabled={!output}>
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Output</CardTitle>
          </CardHeader>
          <CardContent>
            {error && (
              <div className="mb-3 rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </div>
            )}
            <pre className={cn(
              'min-h-[4rem] whitespace-pre-wrap break-all rounded-md bg-muted/50 p-3 text-sm font-mono',
              !output && 'text-muted-foreground'
            )} role="log" aria-live="polite" aria-label="Generation output">
              {output || (engineState === 'generating' ? 'Generating...' : 'Output appears here')}
            </pre>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
