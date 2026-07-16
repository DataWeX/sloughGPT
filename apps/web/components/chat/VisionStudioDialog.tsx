'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { cn, Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Tabs } from '@sloughgpt/strui'
import { IconUpload, IconTrash, IconSend, IconDownload, IconX, IconRefresh } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'

interface VisionStudioDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string | null
  onGeneratedImage: (dataUrl: string, prompt: string) => void
  onSendText: (text: string) => void
  initialCaps?: {
    images_learned?: number
    trained?: boolean
    status?: string
    vocab_size?: number
    mean_accuracy?: number
  }
}

interface AnalyzeResult {
  caption: string
  confidence: number
  tags: string[]
  accuracy: number
  supervised: boolean
  images_learned: number
  trained: boolean
  replay_buffer_size: number
  mean_accuracy: number
}

export function VisionStudioDialog({
  open, onOpenChange, sessionId, onGeneratedImage, onSendText, initialCaps,
}: VisionStudioDialogProps) {
  const [tab, setTab] = useState('analyze')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResult | null>(null)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewFileName, setPreviewFileName] = useState<string | null>(null)
  const [trainLabel, setTrainLabel] = useState('')
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainResult, setTrainResult] = useState<{ caption: string; accuracy: number } | null>(null)
  const [genPrompt, setGenPrompt] = useState('')
  const [genLoading, setGenLoading] = useState(false)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const [trainingReport, setTrainingReport] = useState<{
    images_learned: number
    vocab_size: number
    caption_history: string[]
    accuracy_history: number[]
    mean_accuracy: number
    last_accuracy: number
  } | null>(() => {
    if (initialCaps && (initialCaps.images_learned ?? 0) > 0) {
      return {
        images_learned: initialCaps.images_learned ?? 0,
        vocab_size: initialCaps.vocab_size ?? 0,
        caption_history: [],
        accuracy_history: [],
        mean_accuracy: initialCaps.mean_accuracy ?? 0,
        last_accuracy: initialCaps.mean_accuracy ?? 0,
      }
    }
    return null
  })
  const [resetLoading, setResetLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [retryLoading, setRetryLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)

  const refreshReport = useCallback(async () => {
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const report = await multimodalController.getTrainingReport()
      setTrainingReport({
        images_learned: report.images_learned,
        vocab_size: report.vocab_size,
        caption_history: report.caption_history,
        accuracy_history: report.accuracy_history,
        mean_accuracy: report.mean_accuracy,
        last_accuracy: report.last_accuracy,
      })
    } catch {
      // silent
    }
  }, [])

  useEffect(() => {
    if (open) refreshReport()
  }, [open, refreshReport])

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    setAnalyzeLoading(true)
    setAnalyzeError(null)
    setAnalyzeResult(null)
    setPreviewFileName(file.name)

    const reader = new FileReader()
    reader.onload = () => setPreviewUrl(reader.result as string)
    reader.readAsDataURL(file)

    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.analyzeImage(file)
      setAnalyzeResult(result)
      refreshReport()
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setAnalyzeLoading(false)
    }
  }, [refreshReport])

  const retryAnalyze = useCallback(async () => {
    if (!previewUrl || retryLoading) return
    setRetryLoading(true)
    setAnalyzeError(null)
    setAnalyzeResult(null)
    try {
      const blobRes = await fetch(previewUrl)
      const blob = await blobRes.blob()
      const file = new File([blob], previewFileName || 'retry.png', { type: blob.type || 'image/png' })
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.analyzeImage(file)
      setAnalyzeResult(result)
      refreshReport()
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : 'Retry failed')
    } finally {
      setRetryLoading(false)
    }
  }, [previewUrl, previewFileName, retryLoading, refreshReport])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
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
    setTrainLoading(true)
    setTrainResult(null)
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
    } finally {
      setTrainLoading(false)
    }
  }, [previewUrl, trainLabel, previewFileName, trainLoading, refreshReport])

  const handleGenerateImage = useCallback(async () => {
    if (!genPrompt.trim() || genLoading) return
    setGenLoading(true)
    setGenResult(null)
    setGenError(null)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.generateImage(genPrompt.trim())
      if (result?.image) setGenResult(result.image)
    } catch {
      setGenError('Generation failed')
    } finally {
      setGenLoading(false)
    }
  }, [genPrompt, genLoading])

  const handleSendGeneratedImage = useCallback(() => {
    if (genResult && genPrompt.trim()) {
      onGeneratedImage(genResult, genPrompt.trim())
      onOpenChange(false)
    }
  }, [genResult, genPrompt, onGeneratedImage, onOpenChange])

  const handleReset = useCallback(async () => {
    setResetLoading(true)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      await multimodalController.resetModel()
      setPreviewUrl(null)
      setAnalyzeResult(null)
      setTrainResult(null)
      refreshReport()
    } catch {
      // silent
    } finally {
      setResetLoading(false)
    }
  }, [refreshReport])

  const clearPreview = useCallback(() => {
    setPreviewUrl(null)
    setPreviewFileName(null)
    setAnalyzeResult(null)
    setAnalyzeError(null)
    setTrainResult(null)
  }, [])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90dvh] overflow-hidden flex flex-col p-0 gap-0">
        <DialogHeader className="px-5 pt-4 pb-2 shrink-0">
          <DialogTitle>Vision Studio</DialogTitle>
        </DialogHeader>

        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { value: 'analyze', label: 'Analyze' },
            { value: 'train', label: 'Supervised Train' },
            { value: 'generate', label: 'Generate' },
            { value: 'history', label: 'History' },
          ]}
          className="mx-4"
        />

        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* ── Analyze Tab ── */}
          {tab === 'analyze' && (
            <div className="space-y-4">
              <div
                ref={dropRef}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
                  dragOver ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-primary/40',
                )}
              >
                <IconUpload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm font-medium mb-1">
                  {previewUrl ? 'Click to change image' : 'Drop an image here or click to upload'}
                </p>
                <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>

              {previewUrl && (
                <div className="relative rounded-lg overflow-hidden border border-border/50">
                  <img src={previewUrl} alt="Preview" className="w-full max-h-64 object-contain bg-muted/5" />
                  <button
                    type="button"
                    onClick={clearPreview}
                    className="absolute top-2 right-2 h-7 w-7 flex items-center justify-center rounded-full bg-background/80 hover:bg-background border border-border/50 shadow-sm"
                    aria-label="Clear preview"
                  >
                    <IconX className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}

              {analyzeLoading && (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-4 w-2/3" />
                </div>
              )}

              {analyzeError && (
                <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
                  {analyzeError}
                </div>
              )}

              {analyzeResult && (
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-muted/30 border border-border/40">
                    <div className="text-[10px] text-muted-foreground font-medium mb-1 uppercase tracking-wider">Caption</div>
                    <div className="text-sm leading-relaxed">{analyzeResult.caption}</div>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className="text-lg font-semibold">{analyzeResult.confidence.toFixed(2)}</div>
                      <div className="text-[10px] text-muted-foreground">Confidence</div>
                    </div>
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className="text-lg font-semibold">{analyzeResult.images_learned}</div>
                      <div className="text-[10px] text-muted-foreground">Images learned</div>
                    </div>
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className={cn(
                        'text-lg font-semibold',
                        analyzeResult.mean_accuracy >= 80 ? 'text-success' : analyzeResult.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
                      )}>
                        {analyzeResult.mean_accuracy.toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-muted-foreground">Mean accuracy</div>
                    </div>
                  </div>

                  {analyzeResult.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {analyzeResult.tags.map((tag) => (
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
                      <p className="text-xs text-muted-foreground">
                        The model isn&apos;t confident about this result. Try again — the model may produce a better result on a second pass.
                      </p>
                      <Button size="sm" variant="outline" onClick={retryAnalyze} disabled={retryLoading} className="h-7 text-xs">
                        <IconRefresh className="h-3 w-3 mr-1" />
                        {retryLoading ? 'Retrying…' : 'Retry analysis'}
                      </Button>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onSendText(analyzeResult.caption)}
                      disabled={!analyzeResult.caption || analyzeResult.caption === '[caption failed]'}
                    >
                      <IconSend className="h-3.5 w-3.5 mr-1" />
                      Send caption to chat
                    </Button>
                    {analyzeResult.caption && analyzeResult.caption !== '[caption failed]' && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          navigator.clipboard.writeText(analyzeResult.caption)
                        }}
                      >
                        <IconDownload className="h-3.5 w-3.5 mr-1" />
                        Copy caption
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Train Tab ── */}
          {tab === 'train' && (
            <div className="space-y-4">
              <p className="text-xs text-muted-foreground">
                Train the vision model with supervised labels for better accuracy.
                Provide a ground truth caption to improve the model&apos;s understanding.
              </p>

              <div
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
                  previewUrl ? 'border-primary/30 bg-primary/3' : 'border-border/50 hover:border-primary/40',
                )}
              >
                {previewUrl ? (
                  <div className="relative inline-block">
                    <img src={previewUrl} alt="Training image" className="max-h-40 rounded object-contain" />
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); clearPreview() }}
                      className="absolute -top-2 -right-2 h-5 w-5 flex items-center justify-center rounded-full bg-background border border-border/50 shadow-sm"
                      aria-label="Remove"
                    >
                      <IconX className="h-3 w-3" />
                    </button>
                  </div>
                ) : (
                  <>
                    <IconUpload className="h-6 w-6 mx-auto mb-1 text-muted-foreground" />
                    <p className="text-xs text-muted-foreground">Click to select an image for training</p>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground">Ground truth label (caption)</label>
                <input
                  className="w-full px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  placeholder="e.g., 'a red car on a sunny road'"
                  value={trainLabel}
                  onChange={(e) => setTrainLabel(e.target.value)}
                  disabled={!previewUrl || trainLoading}
                />
              </div>

              <Button
                className="w-full"
                disabled={!previewUrl || trainLoading}
                onClick={handleTrainWithLabel}
              >
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
                  <div className="text-xs text-muted-foreground">
                    Caption: <span className="text-foreground">{trainResult.caption}</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Generate Tab ── */}
          {tab === 'generate' && (
            <div className="space-y-4">
              <div className="flex gap-2">
                <input
                  className="flex-1 px-3 py-2 text-sm border border-input rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  placeholder="Describe an image to generate..."
                  value={genPrompt}
                  onChange={(e) => setGenPrompt(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleGenerateImage() }}
                  disabled={genLoading}
                />
                <Button onClick={handleGenerateImage} disabled={genLoading || !genPrompt.trim()}>
                  {genLoading ? 'Generating...' : 'Generate'}
                </Button>
              </div>

              {genError && (
                <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">{genError}</div>
              )}

              {genResult && (
                <div className="space-y-2">
                  <div className="relative rounded-lg overflow-hidden border border-border/50 bg-muted/5">
                    <img src={genResult} alt="Generated" className="w-full max-h-72 object-contain" />
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={handleSendGeneratedImage}>
                      <IconSend className="h-3.5 w-3.5 mr-1" />
                      Send to chat
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setGenResult(null)}>
                      Dismiss
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── History Tab ── */}
          {tab === 'history' && (
            <div className="space-y-4">
              {!trainingReport && <Skeleton className="h-32" />}

              {trainingReport && (
                <>
                  <div className="grid grid-cols-4 gap-2">
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className="text-lg font-semibold">{trainingReport.images_learned}</div>
                      <div className="text-[10px] text-muted-foreground">Images learned</div>
                    </div>
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className="text-lg font-semibold">{trainingReport.vocab_size}</div>
                      <div className="text-[10px] text-muted-foreground">Vocab size</div>
                    </div>
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className={cn(
                        'text-lg font-semibold',
                        trainingReport.mean_accuracy >= 80 ? 'text-success' : trainingReport.mean_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
                      )}>
                        {trainingReport.mean_accuracy.toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-muted-foreground">Mean accuracy</div>
                    </div>
                    <div className="p-2 rounded bg-muted/30 border border-border/40 text-center">
                      <div className={cn(
                        'text-lg font-semibold',
                        trainingReport.last_accuracy >= 80 ? 'text-success' : trainingReport.last_accuracy >= 50 ? 'text-warning' : 'text-muted-foreground',
                      )}>
                        {trainingReport.last_accuracy.toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-muted-foreground">Last accuracy</div>
                    </div>
                  </div>

                  {trainingReport.accuracy_history.length > 0 && (
                    <div className="space-y-1">
                      <div className="text-xs font-medium text-muted-foreground">Accuracy history</div>
                      <div className="flex items-end gap-0.5 h-16">
                        {trainingReport.accuracy_history.map((a, i) => (
                          <div
                            key={i}
                            title={`${a.toFixed(1)}%`}
                            className={cn(
                              'flex-1 rounded-t transition-all',
                              a >= 80 ? 'bg-success/60' : a >= 50 ? 'bg-warning/60' : 'bg-muted-foreground/30',
                            )}
                            style={{ height: `${Math.max(a * 0.8, 4)}%` }}
                          />
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
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={refreshReport}
                    >
                      <IconRefresh className="h-3.5 w-3.5 mr-1" />
                      Refresh
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-destructive border-destructive/30 hover:bg-destructive/10"
                      onClick={handleReset}
                      disabled={resetLoading}
                    >
                      <IconTrash className="h-3.5 w-3.5 mr-1" />
                      {resetLoading ? 'Resetting...' : 'Reset model'}
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
