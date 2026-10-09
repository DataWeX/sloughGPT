'use client'

import { useState, useCallback } from 'react'
import { cn, ActionCard } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { IconSend, IconBrain, IconCopy, IconX } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'

const OUTPUT_PREVIEW_CHARS = 300
import { generateController } from '@/lib/generate-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'
import type { ModelEntry } from '@/lib/types/models'

interface OutputResult {
  model: string
  text: string
  tokens: number
  elapsedMs: number
  error?: string
}

interface OutputComparisonCardProps {
  models: ModelEntry[]
}

export default function OutputComparisonCard({ models }: OutputComparisonCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [selectedForOutput, setSelectedForOutput] = useState<Set<string>>(new Set())
  const [outputPrompt, setOutputPrompt] = useState('')
  const [outputResults, setOutputResults] = useState<Record<string, OutputResult>>({})
  const [outputLoading, setOutputLoading] = useState(false)
  const [outputExpanded, setOutputExpanded] = useState<Set<string>>(new Set())

  const toggleOutputModel = (id: string) => setSelectedForOutput(prev => {
    const n = new Set(prev)
    if (n.has(id)) n.delete(id); else n.add(id)
    return n
  })

  const runOutputComparison = useCallback(async () => {
    if (!outputPrompt.trim() || selectedForOutput.size < 1) return
    setOutputLoading(true); setOutputResults({})
    try {
      const prompt = outputPrompt.trim()
      const promises = Array.from(selectedForOutput).map(async (modelId) => {
        const reqStart = Date.now()
        try {
          const res = await generateController.generate({ prompt, model: modelId, max_new_tokens: 128 })
          return { model: modelId, text: res.text || '(empty response)', tokens: res.tokens_generated ?? 0, elapsedMs: Date.now() - reqStart } as OutputResult
        } catch (e: unknown) {
          return { model: modelId, text: '', tokens: 0, elapsedMs: Date.now() - reqStart, error: extractErrorMessage(e, String(e)) } as OutputResult
        }
      })
      const results = await Promise.all(promises)
      const map: Record<string, OutputResult> = {}
      for (const r of results) map[r.model] = r
      setOutputResults(map)
    } finally {
      setOutputLoading(false)
    }
  }, [outputPrompt, selectedForOutput])

  const copyOutputResult = (text: string) => {
    navigator.clipboard.writeText(text)
    addToast('Copied to clipboard', 'success')
  }

  const clearOutputComparison = () => {
    setOutputResults({}); setOutputPrompt(''); setOutputExpanded(new Set())
  }

  return (
    <ActionCard
      title="Output Comparison"
      actions={
        Object.keys(outputResults).length > 0 && (
          <Button variant="ghost" size="sm" className="h-6 text-[10px]" onClick={clearOutputComparison}><IconX className="h-2.5 w-2.5 mr-0.5" /> Clear</Button>
        )
      }
      testId="output-comparison"
      contentClassName="space-y-2"
    >
        <Textarea value={outputPrompt} onChange={e => setOutputPrompt(e.target.value)} placeholder="Enter a prompt to compare model outputs..." className="min-h-[60px] text-[11px]" aria-label="Comparison prompt" />
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[10px] text-muted-foreground/60 mr-0.5">Models:</span>
          {models.map(m => <Chip key={m.id} label={m.name} selected={selectedForOutput.has(m.id)} onClick={() => toggleOutputModel(m.id)} />)}
        </div>
        <div className="flex items-center gap-1.5">
          <Button size="sm" className="h-6 text-[10px]" onClick={runOutputComparison} disabled={outputLoading || !outputPrompt.trim() || selectedForOutput.size < 1}>
            {outputLoading ? <><Spinner size="sm" className="mr-0.5" /> Generating…</> : <><IconSend className="h-2.5 w-2.5 mr-0.5" /> Compare</>}
          </Button>
          {outputLoading && <span className="text-[10px] text-muted-foreground/60 animate-pulse" aria-live="polite">Querying {selectedForOutput.size} model{selectedForOutput.size !== 1 ? 's' : ''}…</span>}
        </div>
        {(() => {
          const entries = Object.entries(outputResults)
          return entries.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
            {entries.map(([modelId, r]) => {
              const modelName = models.find(m => m.id === modelId)?.name || modelId
              const isExpanded = outputExpanded.has(modelId)
              const textLen = r.text.length
              const truncated = textLen > OUTPUT_PREVIEW_CHARS && !isExpanded
              return (
                <div key={modelId} className={cn("rounded-lg border p-2 space-y-1.5", r.error ? "border-destructive/30 bg-destructive/5" : "border-border/40 bg-card/50")}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <IconBrain className="h-2.5 w-2.5 text-muted-foreground/60" />
                      <span className="text-[11px] font-medium">{modelName}</span>
                      {r.error ? <Badge label="Error" variant="error" size="sm" /> : <Badge label={`${r.tokens} tok`} variant="default" size="sm" />}
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-[9px] text-muted-foreground/60 tabular-nums">{(r.elapsedMs / 1000).toFixed(1)}s</span>
                      {!r.error && <Button variant="ghost" size="icon-sm" className="h-5 w-5" onClick={() => copyOutputResult(r.text)} aria-label="Copy response"><IconCopy className="h-2.5 w-2.5" /></Button>}
                    </div>
                  </div>
                  {r.error ? <p className="text-[10px] text-destructive">{r.error}</p> : (
                    <div>
                      <p className="text-[10px] leading-relaxed whitespace-pre-wrap">{truncated ? r.text.slice(0, OUTPUT_PREVIEW_CHARS) + '…' : r.text}</p>
                      {truncated && <Button variant="ghost" size="sm" className="h-5 text-[9px] mt-0.5 px-0" onClick={() => setOutputExpanded(prev => { const n = new Set(prev); n.add(modelId); return n })}>Show all ({textLen} chars)</Button>}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )})()}
      </ActionCard>
  )
}
