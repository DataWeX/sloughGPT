'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs } from '@/components/ui/form'
import { IconUpload, IconTrash, IconSend, IconDownload, IconX, IconRefresh, IconCheck } from '@/components/ui'
import { cn } from '@/lib/cn'
import { Skeleton } from '@/components/ui/display'

interface AnalyzeResult {
  caption: string; confidence: number; tags: string[]
  accuracy: number; supervised: boolean; images_learned: number; trained: boolean
  replay_buffer_size: number; mean_accuracy: number
}

export default function VisionPage() {
  const [tab, setTab] = useState('analyze')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(null)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [retryLoading, setRetryLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewFileName, setPreviewFileName] = useState<string | null>(null)
  const [trainLabel, setTrainLabel] = useState('')
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainResult, setTrainResult] = useState<{ caption: string; accuracy: number } | null>(null)
  const [genPrompt, setGenPrompt] = useState('')
  const [genLoading, setGenLoading] = useState(false)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const [vqaPrompt, setVqaPrompt] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [trainingReport, setTrainingReport] = useState<{
    images_learned: number; vocab_size: number; caption_history: string[]
    accuracy_history: number[]; mean_accuracy: number; last_accuracy: number
  } | null>(null)
  const [resetLoading, setResetLoading] = useState(false)
  const [vlmCheckpoints, setVlmCheckpoints] = useState<Array<{
    name: string; path: string; size_mb: number; created_at: string
    soul_name: string; lineage: string; llm: string
    final_loss: number | null; total_steps: number
    mean_accuracy: number | null; description: string
  }>>([])
  const [vlmCkptLoading, setVlmCkptLoading] = useState(false)
  const [vlmCkptLoadName, setVlmCkptLoadName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)

  const refreshReport = useCallback(async () => {
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const report = await multimodalController.getTrainingReport()
      setTrainingReport({
        images_learned: report.images_learned, vocab_size: report.vocab_size,
        caption_history: report.caption_history, accuracy_history: report.accuracy_history,
        mean_accuracy: report.mean_accuracy, last_accuracy: report.last_accuracy,
      })
    } catch {
      // silent
    }
  }, [])

  useEffect(() => { refreshReport() }, [refreshReport])

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    setAnalyzeLoading(true); setAnalyzeError(null); setAnalyzeResult(null)
    setPreviewFileName(file.name)
    const reader = new FileReader()
    reader.onload = () => setPreviewUrl(reader.result as string)
    reader.readAsDataURL(file)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.analyzeImage(file, vqaPrompt.trim() || undefined)
      setAnalyzeResult(result)
      refreshReport()
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : 'Analysis failed')
    } finally { setAnalyzeLoading(false) }
  }, [refreshReport])

  const retryAnalyze = useCallback(async () => {
    if (!previewUrl || retryLoading) return
    setRetryLoading(true); setAnalyzeError(null); setAnalyzeResult(null)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const blobRes = await fetch(previewUrl)
      const blob = await blobRes.blob()
      const file = new File([blob], previewFileName || 'retry.png', { type: blob.type || 'image/png' })
      const result = await multimodalController.analyzeImage(file, vqaPrompt.trim() || undefined)
      setAnalyzeResult(result); refreshReport()
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : 'Retry failed')
    } finally { setRetryLoading(false) }
  }, [previewUrl, previewFileName, retryLoading, refreshReport, vqaPrompt])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) processFile(file)
  }, [processFile])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
    if (e.target) e.target.value = ''
  }, [processFile])

  const handleTrainWithLabel = useCallback(async () => {
    if (!previewUrl || trainLoading) return
    setTrainLoading(true); setTrainResult(null)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const blobRes = await fetch(previewUrl)
      const blob = await blobRes.blob()
      const file = new File([blob], previewFileName || 'upload.png', { type: blob.type || 'image/png' })
      const result = await multimodalController.trainImage(previewUrl, file.name, trainLabel.trim() || undefined)
      setTrainResult({ caption: result.caption, accuracy: result.accuracy })
      refreshReport()
    } catch {
      setTrainResult({ caption: 'Training failed', accuracy: 0 })
    } finally { setTrainLoading(false) }
  }, [previewUrl, trainLabel, previewFileName, trainLoading, refreshReport])

  const handleGenerateImage = useCallback(async () => {
    if (!genPrompt.trim() || genLoading) return
    setGenLoading(true); setGenResult(null); setGenError(null)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.generateImage(genPrompt.trim())
      if (result?.image) setGenResult(result.image)
    } catch { setGenError('Generation failed') }
    finally { setGenLoading(false) }
  }, [genPrompt, genLoading])

  const handleReset = useCallback(async () => {
    setResetLoading(true)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      await multimodalController.resetModel()
      setPreviewUrl(null); setAnalyzeResult(null); setTrainResult(null)
      refreshReport()
    } catch { /* silent */ }
    finally { setResetLoading(false) }
  }, [refreshReport])

  const refreshVlmCheckpoints = useCallback(async () => {
    setVlmCkptLoading(true)
    try {
      const { visualController } = await import('@/lib/visual-controller')
      const res = await visualController.listCheckpoints()
      setVlmCheckpoints(res || [])
    } catch { /* silent */ }
    finally { setVlmCkptLoading(false) }
  }, [])

  const handleVlmLoad = useCallback(async (name: string) => {
    setVlmCkptLoadName(name)
    try {
      const { visualController } = await import('@/lib/visual-controller')
      await visualController.loadCheckpoint(name)
      // Also register as a VLM provider
      await fetch('/visual/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_dir: `models/visual-finetuned` }),
      })
      refreshVlmCheckpoints()
    } catch { /* silent */ }
    finally { setVlmCkptLoadName(null) }
  }, [refreshVlmCheckpoints])

  useEffect(() => { refreshVlmCheckpoints() }, [refreshVlmCheckpoints])

  const clearPreview = useCallback(() => {
    setPreviewUrl(null); setPreviewFileName(null)
    setAnalyzeResult(null); setAnalyzeError(null); setTrainResult(null)
  }, [])

  return (
    <div className="sl-page mx-auto max-w-5xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Vision Studio" />} />
      <div className="space-y-4">
        <Tabs
          value={tab} onChange={setTab}
          tabs={[
            { value: 'analyze', label: 'Analyze' },
            { value: 'train', label: 'Supervised Train' },
            { value: 'generate', label: 'Generate' },
            { value: 'history', label: 'History' },
          ]}
        />

        {/* Analyze Tab */}
        {tab === 'analyze' && (
          <div className="space-y-4">
            <input
              className="w-full px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
              placeholder="Ask a question about the image (optional — leave blank for auto-caption)"
              value={vqaPrompt}
              onChange={e => setVqaPrompt(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && previewUrl && !retryLoading) retryAnalyze() }}
            />
            <div
              ref={dropRef}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                'border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors',
                dragOver ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-primary/40',
              )}
            >
              <IconUpload className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
              <p className="text-sm font-medium mb-1">
                {previewUrl ? 'Click to change image' : 'Drop an image here or click to upload'}
              </p>
              <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP</p>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
            </div>

            {previewUrl && (
              <div className="relative rounded-lg overflow-hidden border border-border/50">
                <img src={previewUrl} alt="Preview" className="w-full max-h-80 object-contain bg-muted/5" />
                <button type="button" onClick={clearPreview}
                  className="absolute top-2 right-2 h-7 w-7 flex items-center justify-center rounded-full bg-background/80 hover:bg-background border border-border/50 shadow-sm"
                  aria-label="Clear preview">
                  <IconX className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {analyzeLoading && (
              <div className="space-y-2"><Skeleton className="h-4 w-3/4" /><Skeleton className="h-4 w-1/2" /><Skeleton className="h-4 w-2/3" /></div>
            )}
            {analyzeError && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">{analyzeError}</div>
            )}

            {analyzeResult && (
              <div className="space-y-3">
                <div className="p-3 rounded-lg bg-muted/30 border border-border/40">
                  <div className="text-[10px] text-muted-foreground font-medium mb-1 uppercase tracking-wider">Caption</div>
                  <div className="text-sm leading-relaxed">{analyzeResult.caption}</div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[{ label: 'Confidence', value: analyzeResult.confidence.toFixed(2), className: '' },
                    { label: 'Images learned', value: String(analyzeResult.images_learned), className: '' },
                    { label: 'Mean accuracy', value: analyzeResult.mean_accuracy.toFixed(1) + '%',
                      className: analyzeResult.mean_accuracy >= 80 ? 'text-success' : analyzeResult.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground' },
                  ].map(s => (
                    <div key={s.label} className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className={'text-lg font-semibold ' + s.className}>{s.value}</div>
                      <div className="text-[10px] text-muted-foreground">{s.label}</div>
                    </div>
                  ))}
                </div>
                {analyzeResult.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {analyzeResult.tags.map(tag => (
                      <span key={tag} className="px-1.5 py-0.5 rounded-full bg-muted/50 text-[10px] border border-border/30">{tag}</span>
                    ))}
                  </div>
                )}
                {(analyzeResult.confidence < 0.3 || analyzeResult.caption === '[caption failed]') && (
                  <div className="p-3 rounded-lg bg-warning/10 border border-warning/20 text-sm space-y-2">
                    <div className="flex items-center gap-2 text-warning font-medium">
                      <svg className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                      Low confidence — consider retrying
                    </div>
                    <p className="text-xs text-muted-foreground">The model isn&apos;t confident about this result. Try again.</p>
                    <Button size="sm" variant="outline" onClick={retryAnalyze} disabled={retryLoading} className="h-7 text-xs">
                      <IconRefresh className="h-3 w-3 mr-1" />{retryLoading ? 'Retrying…' : 'Retry analysis'}
                    </Button>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="outline"
                    onClick={() => { const text = analyzeResult.caption; if (text && text !== '[caption failed]') navigator.clipboard.writeText(text) }}
                    disabled={!analyzeResult.caption || analyzeResult.caption === '[caption failed]'}>
                    <IconDownload className="h-3.5 w-3.5 mr-1" />Copy caption
                  </Button>
                  <Button size="sm" variant="outline"
                    onClick={() => window.open(`/chat?vision_caption=${encodeURIComponent(analyzeResult.caption)}`, '_blank')}>
                    <IconSend className="h-3.5 w-3.5 mr-1" />Open in chat
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Train Tab */}
        {tab === 'train' && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Train the vision model with supervised labels for better accuracy.
            </p>
            <div onClick={() => fileInputRef.current?.click()}
              className={cn('border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
                previewUrl ? 'border-primary/30 bg-primary/3' : 'border-border/50 hover:border-primary/40')}>
              {previewUrl ? (
                <div className="relative inline-block">
                  <img src={previewUrl} alt="Training image" className="max-h-40 rounded object-contain" />
                  <button type="button" onClick={(e) => { e.stopPropagation(); clearPreview() }}
                    className="absolute -top-2 -right-2 h-5 w-5 flex items-center justify-center rounded-full bg-background border border-border/50 shadow-sm" aria-label="Remove">
                    <IconX className="h-3 w-3" />
                  </button>
                </div>
              ) : (
                <><IconUpload className="h-6 w-6 mx-auto mb-1 text-muted-foreground" /><p className="text-xs text-muted-foreground">Click to select an image for training</p></>
              )}
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground">Ground truth label (caption)</label>
              <input className="w-full px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                placeholder="e.g., 'a red car on a sunny road'" value={trainLabel}
                onChange={e => setTrainLabel(e.target.value)} disabled={!previewUrl || trainLoading} />
            </div>
            <Button className="w-full" disabled={!previewUrl || trainLoading} onClick={handleTrainWithLabel}>
              {trainLoading ? 'Training...' : 'Train with label'}
            </Button>
            {trainResult && (
              <div className="p-3 rounded-lg bg-muted/30 border border-border/40 space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">BLEU accuracy:</span>
                  <span className={cn('font-medium', trainResult.accuracy >= 80 ? 'text-success' : trainResult.accuracy >= 50 ? 'text-warning' : 'text-muted-foreground')}>
                    {trainResult.accuracy.toFixed(1)}%
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">Caption: <span className="text-foreground">{trainResult.caption}</span></div>
              </div>
            )}
          </div>
        )}

        {/* Generate Tab */}
        {tab === 'generate' && (
          <div className="space-y-4">
            <div className="flex gap-2">
              <input className="flex-1 px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                placeholder="Describe an image to generate..." value={genPrompt}
                onChange={e => setGenPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') handleGenerateImage() }}
                disabled={genLoading} />
              <Button onClick={handleGenerateImage} disabled={genLoading || !genPrompt.trim()}>
                {genLoading ? 'Generating...' : 'Generate'}
              </Button>
            </div>
            {genError && <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">{genError}</div>}
            {genResult && (
              <div className="space-y-2">
                <div className="relative rounded-lg overflow-hidden border border-border/50 bg-muted/5">
                  <img src={genResult} alt="Generated" className="w-full max-h-72 object-contain" />
                </div>
                <div className="flex gap-2">
                  <Button size="sm" variant="ghost" onClick={() => setGenResult(null)}>Dismiss</Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* History Tab */}
        {tab === 'history' && (
          <div className="space-y-4">
            {!trainingReport && <Skeleton className="h-32" />}
            {trainingReport && (
              <>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { label: 'Images learned', value: String(trainingReport.images_learned), className: '' },
                    { label: 'Vocab size', value: String(trainingReport.vocab_size), className: '' },
                    { label: 'Mean accuracy', value: trainingReport.mean_accuracy.toFixed(1) + '%',
                      className: trainingReport.mean_accuracy >= 80 ? 'text-success' : trainingReport.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground' },
                    { label: 'Last accuracy', value: trainingReport.last_accuracy.toFixed(1) + '%',
                      className: trainingReport.last_accuracy >= 80 ? 'text-success' : trainingReport.last_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground' },
                  ].map(s => (
                    <div key={s.label} className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className={'text-lg font-semibold ' + s.className}>{s.value}</div>
                      <div className="text-[10px] text-muted-foreground">{s.label}</div>
                    </div>
                  ))}
                </div>
                {trainingReport.accuracy_history.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-muted-foreground">Accuracy history</div>
                    <div className="flex items-end gap-0.5 h-16">
                      {trainingReport.accuracy_history.map((a, i) => (
                        <div key={i} title={`${a.toFixed(1)}%`}
                          className={cn('flex-1 rounded-t transition-all',
                            a >= 80 ? 'bg-success/60' : a >= 50 ? 'bg-warning/60' : 'bg-muted-foreground/30')}
                          style={{ height: `${Math.max(a * 0.8, 4)}%` }} />
                      ))}
                    </div>
                  </div>
                )}
                {trainingReport.caption_history.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-muted-foreground">Recent captions learned</div>
                    <ul className="space-y-1 max-h-40 overflow-y-auto">
                      {trainingReport.caption_history.slice(-20).reverse().map((cap, i) => (
                        <li key={i} className="p-2 rounded bg-muted/20 border border-border/20 text-xs leading-relaxed">{cap}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline" onClick={refreshReport}>
                    <IconRefresh className="h-3.5 w-3.5 mr-1" />Refresh
                  </Button>
                  <Button size="sm" variant="outline" className="text-destructive border-destructive/30 hover:bg-destructive/10"
                    onClick={handleReset} disabled={resetLoading}>
                    <IconTrash className="h-3.5 w-3.5 mr-1" />{resetLoading ? 'Resetting...' : 'Reset model'}
                  </Button>
                </div>
              </>
            )}
          </div>
        )}

        {/* VLM Checkpoints */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center justify-between">
              <span>VLM Checkpoints</span>
              <Button size="sm" variant="ghost" onClick={refreshVlmCheckpoints} disabled={vlmCkptLoading} className="h-7 text-xs">
                <IconRefresh className="h-3 w-3 mr-1" />Refresh
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {vlmCkptLoading && vlmCheckpoints.length === 0 && (
              <div className="space-y-2"><Skeleton className="h-8" /><Skeleton className="h-8" /><Skeleton className="h-8" /></div>
            )}
            {!vlmCkptLoading && vlmCheckpoints.length === 0 && (
              <div className="py-6 text-center text-xs text-muted-foreground">
                No VLM checkpoints found. Train a model on the{" "}
                <a href="/multimodal" className="underline hover:text-primary">Multimodal</a> page.
              </div>
            )}
            {vlmCheckpoints.length > 0 && (
              <div className="space-y-1">
                {vlmCheckpoints.map(cp => (
                  <div key={cp.name} className="flex items-center justify-between p-2 rounded bg-muted/20 border border-border/20 text-sm">
                    <div className="min-w-0">
                      <div className="font-medium truncate">{cp.soul_name || cp.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {cp.llm} · {cp.size_mb.toFixed(1)} MB
                        {cp.total_steps > 0 && ` · ${cp.total_steps} steps`}
                        {cp.final_loss != null && ` · loss ${cp.final_loss.toFixed(3)}`}
                        {cp.mean_accuracy != null && ` · acc ${cp.mean_accuracy.toFixed(1)}%`}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Button
                        size="sm" variant="ghost" className="h-7 text-xs"
                        onClick={() => handleVlmLoad(cp.name)}
                        disabled={vlmCkptLoadName === cp.name}
                      >
                        {vlmCkptLoadName === cp.name ? (
                          <><IconCheck className="h-3 w-3 mr-1" />Loading…</>
                        ) : 'Load'}
                      </Button>
                      <Button
                        size="sm" variant="ghost" className="h-7 text-xs text-destructive hover:text-destructive"
                        onClick={async () => {
                          const { visualController } = await import('@/lib/visual-controller')
                          await visualController.deleteCheckpoint(cp.name)
                          refreshVlmCheckpoints()
                        }}
                      >
                        <IconTrash className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
