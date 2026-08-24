'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { modelController, type QuantizationResult } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

interface QuantizeCardProps {
  isLoaded: boolean
  modelId: string
  health: { quantization?: { quantized: boolean; bits?: number; mode?: string; summary?: { bits: number; tensors: number; avg_cosine_sim: number; min_cosine_sim: number } } } | null
  onQuantized?: () => void
}

interface DisplayState {
  quantized: boolean
  bits?: number
  mode?: string
  tensors?: number
  avg_cosine_sim?: number
  min_cosine_sim?: number
}

export function QuantizeCard({ isLoaded, modelId, health, onQuantized }: QuantizeCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [quantizing, setQuantizing] = useState(false)
  const [dequantizing, setDequantizing] = useState(false)
  const [lastResult, setLastResult] = useState<QuantizationResult | null>(null)

  const q = health?.quantization
  const isQuantized = q?.quantized ?? false

  const handleQuantize = useCallback(async (bits: number) => {
    if (quantizing || dequantizing) return
    setQuantizing(true)
    try {
      const result = await modelController.quantize(bits, 'symmetric')
      setLastResult(result)
      addToast(`Quantized to ${bits}-bit (${result.summary?.tensors ?? 0} tensors)`, 'success')
      onQuantized?.()
    } catch {
      addToast('Could not quantization', 'error')
    } finally {
      setQuantizing(false)
    }
  }, [quantizing, dequantizing, addToast, onQuantized])

  const handleDequantize = useCallback(async () => {
    if (quantizing || dequantizing) return
    setDequantizing(true)
    try {
      await modelController.dequantize()
      setLastResult(null)
      addToast('Model restored to full precision', 'success')
      onQuantized?.()
    } catch {
      addToast('Could not dequantization', 'error')
    } finally {
      setDequantizing(false)
    }
  }, [quantizing, dequantizing, addToast, onQuantized])

  if (!isLoaded) return null

  const display: DisplayState | null = lastResult
    ? { quantized: lastResult.quantized, bits: lastResult.bits, mode: lastResult.mode, tensors: lastResult.summary?.tensors, avg_cosine_sim: lastResult.summary?.avg_cosine_sim, min_cosine_sim: lastResult.summary?.min_cosine_sim }
    : q
      ? { quantized: q.quantized, bits: q.bits, mode: q.mode, tensors: q.summary?.tensors, avg_cosine_sim: q.summary?.avg_cosine_sim, min_cosine_sim: q.summary?.min_cosine_sim }
      : null

  return (
    <Card data-testid="quantize-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Quantize</CardTitle>
            {isQuantized && (
              <Badge
                label={`${q?.bits ?? '?'}-bit`}
                variant="success"
                size="sm"
              />
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              onClick={() => handleQuantize(8)}
              disabled={quantizing || dequantizing}
            >
              {quantizing ? 'Working…' : 'Int8'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs"
              onClick={() => handleQuantize(4)}
              disabled={quantizing || dequantizing}
            >
              {quantizing ? 'Working…' : 'Int4'}
            </Button>
            {isQuantized && (
              <Button
                size="sm"
                variant="ghost"
                className="h-7 text-xs"
                onClick={handleDequantize}
                disabled={quantizing || dequantizing}
              >
                {dequantizing ? 'Working…' : 'Restore'}
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!display ? (
          <p className="text-sm text-muted-foreground text-center py-2">
            Quantize to reduce memory at the cost of slight quality loss.
          </p>
        ) : (
          <div className="space-y-2 text-[11px]">
            {display.bits != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Precision</span>
                <span className="font-mono">{display.bits}-bit {display.mode || 'symmetric'}</span>
              </div>
            )}
            {display.tensors != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Tensors</span>
                <span className="font-mono">{display.tensors}</span>
              </div>
            )}
            {display.avg_cosine_sim != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg quality</span>
                <span className="font-mono">{display.avg_cosine_sim.toFixed(3)}</span>
              </div>
            )}
            {display.min_cosine_sim != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Min quality</span>
                <span className="font-mono">{display.min_cosine_sim.toFixed(3)}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
