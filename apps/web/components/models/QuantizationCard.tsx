'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { modelController, type QuantizationResult } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

type TensorEntry = { scale: number; zero_point: number; cosine_sim: number }

function cosineColor(cos: number): string {
  if (cos >= 0.99) return 'text-success'
  if (cos >= 0.95) return 'text-warning'
  return 'text-destructive'
}

function formatLayerName(name: string): string {
  return name.replace('.weight', '').replace('blocks.', 'B').replace('.q_proj', '/Q').replace('.k_proj', '/K').replace('.v_proj', '/V').replace('.o_proj', '/O').replace('.fc1', '/FC1').replace('.fc2', '/FC2')
}

function qualityGrade(cos: number): { label: string; color: string } {
  if (cos >= 0.999) return { label: 'Excellent', color: 'text-success' }
  if (cos >= 0.99) return { label: 'Good', color: 'text-success' }
  if (cos >= 0.95) return { label: 'Fair', color: 'text-warning' }
  return { label: 'Poor', color: 'text-destructive' }
}

export default function QuantizationCard({ isOnline }: { isOnline: boolean }) {
  const addToast = useToastStore(s => s.addToast)
  const [bits, setBits] = useState<4 | 8>(8)
  const [mode, setMode] = useState<'symmetric' | 'asymmetric'>('symmetric')
  const [quantizing, setQuantizing] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [result, setResult] = useState<QuantizationResult | null>(null)
  const [showLayers, setShowLayers] = useState(false)

  const handleQuantize = async () => {
    setQuantizing(true)
    setResult(null)
    try {
      const res = await modelController.quantize(bits, mode)
      setResult(res)
      addToast(`${res.bits}-bit ${mode} quantization applied (${res.layers_quantized} layers)`, 'success')
    } catch (err) {
      addToast(extractErrorMessage(err, 'Quantization failed'), 'error')
    } finally {
      setQuantizing(false)
    }
  }

  const handleReset = async () => {
    setResetting(true)
    try {
      await modelController.dequantize()
      setResult(null)
      setShowLayers(false)
      addToast('Reset to float32', 'success')
    } catch (err) {
      addToast(extractErrorMessage(err, 'Reset failed'), 'error')
    } finally {
      setResetting(false)
    }
  }

  const sortedLayers = result
    ? (Object.entries(result.per_tensor) as [string, TensorEntry][])
        .sort((a, b) => a[1].cosine_sim - b[1].cosine_sim)
    : []

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Quantization</CardTitle></CardHeader>
      <CardContent>
        {!isOnline ? (
          <p className="text-sm text-muted-foreground">Load a model first to quantize</p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5" role="radiogroup" aria-label="Quantization precision">
                <span className="text-xs text-muted-foreground">Precision:</span>
                <button
                  type="button"
                  onClick={() => setBits(8)}
                  aria-pressed={bits === 8}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${bits === 8 ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  int8
                </button>
                <button
                  type="button"
                  onClick={() => setBits(4)}
                  aria-pressed={bits === 4}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${bits === 4 ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  int4
                </button>
              </div>
              <div className="flex items-center gap-1.5" role="radiogroup" aria-label="Quantization mode">
                <span className="text-xs text-muted-foreground">Mode:</span>
                <button
                  type="button"
                  onClick={() => setMode('symmetric')}
                  aria-pressed={mode === 'symmetric'}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${mode === 'symmetric' ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  Sym
                </button>
                <button
                  type="button"
                  onClick={() => setMode('asymmetric')}
                  aria-pressed={mode === 'asymmetric'}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${mode === 'asymmetric' ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  Asym
                </button>
              </div>
              <Button size="sm" disabled={quantizing} onClick={handleQuantize}>
                {quantizing ? 'Quantizing...' : 'Apply'}
              </Button>
              {result && (
                <Button size="sm" variant="outline" disabled={resetting} onClick={handleReset}>
                  {resetting ? 'Resetting...' : 'Reset'}
                </Button>
              )}
            </div>

            {result && (() => {
              const grade = qualityGrade(result.summary.avg_cosine_sim)
              return (
                <div className="pt-2 border-t border-border/40 space-y-2">
                  <KpiGrid columns={4}>
                    <StatCard label="Layers" value={`${result.layers_quantized}/${result.total_layers}`} />
                    <StatCard label="Quality" value={grade.label} icon={<span className={`text-xs ${grade.color}`}>●</span>} />
                    <StatCard label="Mode" value={result.mode === 'asymmetric' ? 'Asym' : 'Sym'} />
                    <StatCard label="Type" value={result.model_type === 'slonet' ? 'SloNet' : 'HuggingFace'} />
                  </KpiGrid>

                {sortedLayers.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setShowLayers(!showLayers)}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showLayers ? 'Hide' : 'Show'} per-layer detail ({sortedLayers.length} layers, sorted by quality)
                  </button>
                )}

                {showLayers && (
                  <div className="max-h-64 overflow-x-auto overflow-y-auto rounded border border-border/40">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-muted/50">
                        <tr className="text-left text-muted-foreground">
                          <th className="px-2 py-1 font-medium">Layer</th>
                          <th className="px-2 py-1 font-medium text-right">Cosine</th>
                          <th className="px-2 py-1 font-medium text-right">Scale</th>
                          <th className="px-2 py-1 font-medium text-right">Zero Pt</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedLayers.map(([name, entry]) => (
                          <tr key={name} className="border-t border-border/20 hover:bg-muted/20 transition-colors">
                            <td className="px-2 py-1 font-mono">{formatLayerName(name)}</td>
                            <td className={`px-2 py-1 text-right font-mono ${cosineColor(entry.cosine_sim)}`}>
                              {entry.cosine_sim.toFixed(6)}
                            </td>
                            <td className="px-2 py-1 text-right font-mono">{entry.scale.toFixed(6)}</td>
                            <td className="px-2 py-1 text-right font-mono">{entry.zero_point}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )})()}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
