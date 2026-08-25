'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { useToastStore } from '@/lib/toast-store'
import {
  worldController,
  type RenderConfig,
  type TickResult,
  type NeuralResult,
  type WorldStats,
} from '@/lib/world-controller'

export default function WorldPage() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<WorldStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [ticking, setTicking] = useState(false)

  const [config, setConfig] = useState<RenderConfig>({
    width: 160,
    height: 120,
    samples: 16,
    camera_height: 40,
    camera_distance: 30,
  })

  const [renderResult, setRenderResult] = useState<{ shapes: Record<string, number[]>; tensorKeys: string[] } | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [tickResult, setTickResult] = useState<TickResult | null>(null)
  const [neuralResult, setNeuralResult] = useState<NeuralResult | null>(null)
  const [tickCount, setTickCount] = useState(0)
  const imageUrlRef = useRef<string | null>(null)

  const loadStats = useCallback(async () => {
    try {
      const s = await worldController.stats()
      setStats(s)
    } catch {
      // silently fail
    }
  }, [])

  useEffect(() => { void loadStats() }, [loadStats])

  const handleRender = async () => {
    setRendering(true)
    setRenderResult(null)
    if (imageUrlRef.current) { URL.revokeObjectURL(imageUrlRef.current); imageUrlRef.current = null; setImageUrl(null) }
    try {
      const result = await worldController.render(config)
      setRenderResult({ shapes: result.shapes, tensorKeys: result.tensor_keys })
      const blob = await worldController.renderImage(config)
      const url = URL.createObjectURL(blob)
      imageUrlRef.current = url
      setImageUrl(url)
      addToast('Render complete', 'success')
    } catch (e) {
      addToast(`Render failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setRendering(false)
    }
  }

  const handleTick = async (neural = false) => {
    setTicking(true)
    setTickResult(null)
    setNeuralResult(null)
    try {
      const result = await worldController.tick(1, true, neural)
      setTickResult(result)
      setTickCount(c => c + 1)
      if (neural) {
        const n = await worldController.neuralProcess(config)
        setNeuralResult(n)
      }
      addToast(`Tick ${result.tick} complete — ${result.babies} babies`, 'success')
    } catch (e) {
      addToast(`Tick failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error')
    } finally {
      setTicking(false)
    }
  }

  return (
    <PageContainer
      title="World Render"
      subtitle="Programmable world simulation and rendering"
      headerRight={
        <Button size="sm" variant="ghost" onClick={() => void loadStats()}>Refresh</Button>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {stats?.components.map(c => (
            <div key={c} className="rounded-md bg-muted/30 p-3 text-center">
              <div className="text-[10px] text-muted-foreground">Component</div>
              <div className="text-xs font-medium">{c}</div>
            </div>
          ))}
          <div className="rounded-md bg-muted/30 p-3 text-center">
            <div className="text-[10px] text-muted-foreground">Ticks Run</div>
            <div className="text-xs font-medium">{tickCount}</div>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Render Config</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <div className="space-y-1">
                <Label className="text-xs">Width</Label>
                <Input type="number" value={config.width ?? 160} onChange={e => setConfig(c => ({ ...c, width: Number(e.target.value) }))} className="h-8 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Height</Label>
                <Input type="number" value={config.height ?? 120} onChange={e => setConfig(c => ({ ...c, height: Number(e.target.value) }))} className="h-8 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Samples</Label>
                <Input type="number" value={config.samples ?? 16} onChange={e => setConfig(c => ({ ...c, samples: Number(e.target.value) }))} className="h-8 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Camera Height</Label>
                <Input type="number" step="0.5" value={config.camera_height ?? 40} onChange={e => setConfig(c => ({ ...c, camera_height: Number(e.target.value) }))} className="h-8 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Camera Distance</Label>
                <Input type="number" step="0.5" value={config.camera_distance ?? 30} onChange={e => setConfig(c => ({ ...c, camera_distance: Number(e.target.value) }))} className="h-8 text-xs" />
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => void handleRender()} disabled={rendering} className="flex-1">
                {rendering ? 'Rendering...' : 'Render'}
              </Button>
              <Button onClick={() => void handleTick(false)} disabled={ticking} variant="outline" className="flex-1">
                {ticking ? 'Ticking...' : 'Run Tick'}
              </Button>
              <Button onClick={() => void handleTick(true)} disabled={ticking} variant="outline" className="flex-1">
                {ticking ? 'Processing...' : 'Tick + Neural'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {imageUrl && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Render Output</CardTitle>
            </CardHeader>
            <CardContent>
              <img src={imageUrl} alt="World render" loading="lazy" className="w-full rounded border border-border" />
            </CardContent>
          </Card>
        )}

        {renderResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">State Tensors</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {renderResult.tensorKeys.map(k => (
                  <div key={k} className="rounded bg-muted/30 p-2">
                    <div className="text-[10px] text-muted-foreground">{k}</div>
                    <div className="text-xs font-mono">{String(renderResult.shapes[k])}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {tickResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tick Result</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <div className="rounded bg-muted/30 p-3 text-center">
                  <div className="text-[10px] text-muted-foreground">Tick</div>
                  <div className="text-lg font-mono font-medium">{tickResult.tick}</div>
                </div>
                <div className="rounded bg-muted/30 p-3 text-center">
                  <div className="text-[10px] text-muted-foreground">Babies</div>
                  <div className="text-lg font-mono font-medium">{tickResult.babies}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {neuralResult && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Neural Processing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded bg-muted/30 p-3 text-center">
                  <div className="text-[10px] text-muted-foreground">Embedding Shape</div>
                  <div className="text-xs font-mono">{neuralResult.embedding_shape ? String(neuralResult.embedding_shape) : 'N/A'}</div>
                </div>
              </div>
              <pre className="max-h-[200px] overflow-y-auto rounded bg-muted/30 p-3 text-xs">{JSON.stringify(neuralResult.descriptor, null, 2)}</pre>
            </CardContent>
          </Card>
        )}

        {stats && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Materials</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {Object.entries(stats.materials).map(([name, id]) => (
                  <span key={name} className="inline-flex items-center gap-1.5 rounded border border-border bg-muted/30 px-2 py-1 text-xs">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: ['#94a3b8', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#06b6d4'][id] ?? '#94a3b8' }} />
                    {name} ({id})
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
