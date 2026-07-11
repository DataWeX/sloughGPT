'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { KpiGrid, StatCard } from '@sloughgpt/strui'
import { modelController, type QuantizationResult } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

export default function QuantizationCard({ isOnline }: { isOnline: boolean }) {
  const addToast = useToastStore(s => s.addToast)
  const [bits, setBits] = useState<4 | 8>(8)
  const [quantizing, setQuantizing] = useState(false)
  const [result, setResult] = useState<QuantizationResult | null>(null)

  const handleQuantize = async () => {
    setQuantizing(true)
    setResult(null)
    try {
      const res = await modelController.quantize(bits, 'symmetric')
      setResult(res)
      addToast(`${res.bits}-bit quantization applied (${res.layers_quantized} layers)`, 'success')
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Quantization failed', 'error')
    } finally {
      setQuantizing(false)
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Quantization</CardTitle></CardHeader>
      <CardContent>
        {!isOnline ? (
          <p className="text-sm text-muted-foreground">Load a model first to quantize</p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Precision:</span>
                <button
                  type="button"
                  onClick={() => setBits(8)}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${bits === 8 ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  int8
                </button>
                <button
                  type="button"
                  onClick={() => setBits(4)}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${bits === 4 ? 'bg-primary/10 border-primary text-primary' : 'border-border/60 text-muted-foreground hover:border-primary/40'}`}
                >
                  int4
                </button>
              </div>
              <Button size="sm" disabled={quantizing} onClick={handleQuantize}>
                {quantizing ? 'Quantizing...' : 'Apply'}
              </Button>
            </div>

            {result && (
              <div className="pt-2 border-t border-border/40">
                <KpiGrid columns={3}>
                  <StatCard label="Layers" value={`${result.layers_quantized}/${result.total_layers}`} />
                  <StatCard label="Avg Cosine" value={result.summary.avg_cosine_sim.toFixed(4)} />
                  <StatCard label="AVX2" value={result.avx2_enabled ? 'Enabled' : 'N/A'} />
                </KpiGrid>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
