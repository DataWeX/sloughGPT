'use client'

import { useState, useCallback } from 'react'
import { cn, Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { IconSend, IconBrain, IconCopy, IconX } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'
import { generateController } from '@/lib/generate-controller'
import { useToastStore } from '@/lib/toast-store'

interface OutputResult {
  model: string
  text: string
  tokens: number
  elapsedMs: number
  error?: string
}

interface OutputComparisonCardProps {
  models: { id: string; name: string }[]
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
    const prompt = outputPrompt.trim()
    const promises = Array.from(selectedForOutput).map(async (modelId) => {
      const reqStart = Date.now()
      try {
        const res = await generateController.generate({ prompt, model: modelId, max_new_tokens: 128 })
        return { model: modelId, text: res.text || '(empty response)', tokens: res.tokens_generated ?? 0, elapsedMs: Date.now() - reqStart } as OutputResult
      } catch (e: any) {
        return { model: modelId, text: '', tokens: 0, elapsedMs: Date.now() - reqStart, error: String(e?.message || e) } as OutputResult
      }
    })
    const results = await Promise.all(promises)
    const map: Record<string, OutputResult> = {}
    for (const r of results) map[r.model] = r
    setOutputResults(map); setOutputLoading(false)
  }, [outputPrompt, selectedForOutput])

  const copyOutputResult = (text: string) => {
    navigator.clipboard.writeText(text)
    addToast('Copied to clipboard', 'success')
  }

  const clearOutputComparison = () => {
    setOutputResults({}); setOutputPrompt(''); setOutputExpanded(new Set())
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Output Comparison</CardTitle>
          {Object.keys(outputResults).length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearOutputComparison}><IconX className="h-3.5 w-3.5 mr-1" /> Clear</Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea value={outputPrompt} onChange={e => setOutputPrompt(e.target.value)} placeholder="Enter a prompt to compare model outputs..." className="min-h-[80px] text-sm" aria-label="Comparison prompt" />
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted-foreground mr-1">Models:</span>
          {models.map(m => <Chip key={m.id} label={m.name} selected={selectedForOutput.has(m.id)} onClick={() => toggleOutputModel(m.id)} />)}
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={runOutputComparison} disabled={outputLoading || !outputPrompt.trim() || selectedForOutput.size < 1}>
            {outputLoading ? <><Spinner size="sm" className="mr-1" /> Generating…</> : <><IconSend className="h-3.5 w-3.5 mr-1" /> Compare</>}
          </Button>
          {outputLoading && <span className="text-xs text-muted-foreground animate-pulse">Querying {selectedForOutput.size} model{selectedForOutput.size !== 1 ? 's' : ''}…</span>}
        </div>
        {Object.keys(outputResults).length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            {Object.entries(outputResults).map(([modelId, r]) => {
              const modelName = models.find(m => m.id === modelId)?.name || modelId
              const isExpanded = outputExpanded.has(modelId)
              const textLen = r.text.length
              const truncated = textLen > 300 && !isExpanded
              return (
                <div key={modelId} className={cn("rounded-lg border p-3 space-y-2", r.error ? "border-destructive/30 bg-destructive/5" : "border-border/60 bg-card/50")}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <IconBrain className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-sm font-medium">{modelName}</span>
                      {r.error ? <Badge label="Error" variant="error" size="sm" /> : <Badge label={`${r.tokens} tok`} variant="default" size="sm" />}
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-muted-foreground">{(r.elapsedMs / 1000).toFixed(1)}s</span>
                      {!r.error && <Button variant="ghost" size="icon-sm" className="h-6 w-6" onClick={() => copyOutputResult(r.text)} aria-label="Copy response"><IconCopy className="h-3 w-3" /></Button>}
                    </div>
                  </div>
                  {r.error ? <p className="text-xs text-destructive">{r.error}</p> : (
                    <div>
                      <p className="text-xs leading-relaxed whitespace-pre-wrap">{truncated ? r.text.slice(0, 300) + '…' : r.text}</p>
                      {truncated && <Button variant="ghost" size="sm" className="h-6 text-xs mt-1 px-0" onClick={() => setOutputExpanded(prev => { const n = new Set(prev); n.add(modelId); return n })}>Show all ({textLen} chars)</Button>}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
