'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { StatCard, KpiGrid, Skeleton, ProgressBar } from '@/components/ui'
import { IconRefresh, IconUpload, IconTrash, IconCheck } from '@/components/ui'
import { cn } from '@/lib/cn'
import { multimodalController, type MultimodalCapabilities, type TrainingReport, type TrainingStatus } from '@/lib/multimodal-controller'
import { useToastStore } from '@/lib/toast-store'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function MultimodalPage() {
  const addToast = useToastStore(s => s.addToast)
  const [caps, setCaps] = useState<MultimodalCapabilities | null>(null)
  const [report, setReport] = useState<TrainingReport | null>(null)
  const [trainStatus, setTrainStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [batchUploading, setBatchUploading] = useState(false)
  const [batchDirPath, setBatchDirPath] = useState('')
  const [vlmDatasetName, setVlmDatasetName] = useState('')
  const [vlmImageDir, setVlmImageDir] = useState('')
  const [creatingDataset, setCreatingDataset] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [genPrompt, setGenPrompt] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [transcript, setTranscript] = useState<string | null>(null)
  const [synthesizing, setSynthesizing] = useState(false)
  const [synthText, setSynthText] = useState('')
  const [vlmInferring, setVlmInferring] = useState(false)
  const [vlmOutput, setVlmOutput] = useState<string | null>(null)
  const [vlmPrompt, setVlmPrompt] = useState('Describe this image in detail.')
  const [vlmLoaded, setVlmLoaded] = useState(false)
  const vlmImageInputRef = useRef<HTMLInputElement>(null)
  const [vlmImageBase64, setVlmImageBase64] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const batchFileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [c, r, s, vlmStatus] = await Promise.all([
        multimodalController.getCapabilities(),
        multimodalController.getTrainingReport().catch(() => null),
        multimodalController.getTrainingStatus().catch(() => null),
        multimodalController.getVLMStatus().catch(() => ({ loaded: false })),
      ])
      setCaps(c)
      setReport(r)
      setTrainStatus(s)
      setVlmLoaded(vlmStatus.loaded)
    } catch {
      addToast('Failed to load multimodal data', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (trainStatus?.running && !pollIntervalRef.current) {
      startPolling()
    }
  }, [trainStatus?.running])

  const handleUploadImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      const reader = new FileReader()
      reader.onload = async () => {
        const dataUrl = reader.result as string
        const result = await multimodalController.trainImage(dataUrl, file.name)
        addToast(`Trained on "${file.name}" — caption: ${result.caption}`, 'success')
        fetchAll()
      }
      reader.readAsDataURL(file)
    } catch {
      addToast('Upload failed', 'error')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const startPolling = () => {
    if (pollIntervalRef.current) return
    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await multimodalController.getTrainingStatus()
        setTrainStatus(status)
        if (!status.running) {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
          fetchAll()
          if (status.completed > 0) {
            addToast(`Batch training complete: ${status.completed} images, ${status.errors} errors`, status.errors > 0 ? 'error' : 'success')
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 2000)
  }

  const handleBatchUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatch(files)
      addToast(`Batch training started: ${result.total_images} images`, 'success')
      startPolling()
    } catch {
      addToast('Batch upload failed', 'error')
    } finally {
      setBatchUploading(false)
      if (batchFileInputRef.current) batchFileInputRef.current.value = ''
    }
  }

  const handleBatchDir = async () => {
    if (!batchDirPath.trim()) return
    setBatchUploading(true)
    try {
      const result = await multimodalController.trainBatchFromDir(batchDirPath.trim())
      addToast(`Batch training started: ${result.total_images} images from ${batchDirPath}`, 'success')
      startPolling()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Batch training failed'
      addToast(msg, 'error')
    } finally {
      setBatchUploading(false)
    }
  }

  const handleCreateVLMDataset = async () => {
    if (!vlmDatasetName.trim() || !vlmImageDir.trim()) return
    setCreatingDataset(true)
    try {
      const result = await multimodalController.createVLMDataset(
        vlmDatasetName.trim(),
        vlmImageDir.trim(),
      )
      addToast(`Dataset "${result.dataset}" created: ${result.entries} entries`, 'success')
      setVlmDatasetName('')
      setVlmImageDir('')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Dataset creation failed'
      addToast(msg, 'error')
    } finally {
      setCreatingDataset(false)
    }
  }

  const handleGenerateImage = async () => {
    if (!genPrompt.trim()) return
    setGenerating(true)
    setGeneratedImage(null)
    try {
      const result = await multimodalController.generateImage(genPrompt)
      setGeneratedImage(result.image)
      addToast(`Generated: "${result.prompt}"`, 'success')
    } catch {
      addToast('Image generation failed', 'error')
    } finally {
      setGenerating(false)
    }
  }

  const handleTranscribe = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setTranscribing(true)
    setTranscript(null)
    try {
      const result = await multimodalController.transcribeAudio(file)
      setTranscript(result.text)
    } catch {
      addToast('Transcription failed', 'error')
    } finally {
      setTranscribing(false)
      if (audioInputRef.current) audioInputRef.current.value = ''
    }
  }

  const handleSynthesize = async () => {
    if (!synthText.trim()) return
    setSynthesizing(true)
    try {
      const result = await multimodalController.synthesizeSpeech(synthText)
      addToast(`Speech synthesized (${result.duration_sec.toFixed(1)}s)`, 'success')
    } catch {
      addToast('Speech synthesis failed', 'error')
    } finally {
      setSynthesizing(false)
    }
  }

  const handleVLMImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      setVlmImageBase64(dataUrl)
      addToast(`Selected "${file.name}"`, 'info')
    }
    reader.readAsDataURL(file)
  }

  const handleVLMInfer = async () => {
    if (!vlmImageBase64) { addToast('Select an image first', 'error'); return }
    setVlmInferring(true)
    setVlmOutput(null)
    try {
      const base64 = vlmImageBase64.split(',')[1] || vlmImageBase64
      const result = await multimodalController.vlmInference(base64, vlmPrompt)
      setVlmOutput(result.text)
      addToast(`Generated ${result.tokens_generated} tokens in ${(result.elapsed_ms / 1000).toFixed(1)}s`, 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'VLM inference failed'
      addToast(msg, 'error')
    } finally {
      setVlmInferring(false)
    }
  }

  const capList: { label: string; ok: boolean }[] = caps ? [
    { label: 'Speech-to-text', ok: caps.speech_to_text },
    { label: 'Image captioning', ok: caps.image_caption },
    { label: 'Vision model', ok: !!caps.vision_model },
    { label: 'Speech model', ok: !!caps.speech_model },
    { label: 'Trained', ok: caps.trained },
    { label: 'VLM loaded', ok: vlmLoaded },
  ] : []

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Multimodal" subtitle="Vision, speech, and image generation" />}
        right={
          <Button variant="outline" size="sm" onClick={fetchAll} disabled={loading}>
            <IconRefresh className="h-3.5 w-3.5 mr-1" /> Refresh
          </Button>
        }
      />

      <div className="space-y-4">
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-24 rounded-lg" />
            <Skeleton className="h-48 rounded-lg" />
          </div>
        ) : (
          <>
            {/* Overview */}
            <Card>
              <CardHeader><CardTitle className="text-base">Capabilities</CardTitle></CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mb-4">
                  {capList.map(c => (
                    <Badge
                      key={c.label}
                      label={c.label}
                      variant={c.ok ? 'success' : 'warning'}
                      size="sm"
                    />
                  ))}
                </div>
                <KpiGrid columns={4}>
                  <StatCard label="Images learned" value={caps?.images_learned ?? 0} />
                  <StatCard label="Replay buffer" value={`${caps?.replay_buffer_size ?? 0} items`} />
                  <StatCard label="Learning method" value={caps?.learning_method || '—'} />
                  <StatCard label="Status" value={caps?.status || '—'} />
                </KpiGrid>
              </CardContent>
            </Card>

            {/* Training progress */}
            {report && (
              <Card>
                <CardHeader><CardTitle className="text-base">Training</CardTitle></CardHeader>
                <CardContent>
                  <KpiGrid columns={4}>
                    <StatCard label="Images learned" value={report.images_learned} />
                    <StatCard label="Vocabulary" value={`${report.vocab_size} words`} />
                    <StatCard label="Unique captions" value={report.unique_captions} />
                    <StatCard label="Diversity" value={`${(report.diversity_ratio * 100).toFixed(0)}%`} />
                  </KpiGrid>
                  {trainStatus && trainStatus.total > 0 && (
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{trainStatus.completed}/{trainStatus.total} images</span>
                        <span>{trainStatus.progress_pct}%</span>
                      </div>
                      <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
                      {trainStatus.current_caption && (
                        <p className="text-[10px] text-muted-foreground/60 truncate">
                          Current: {trainStatus.current_caption}
                        </p>
                      )}
                    </div>
                  )}
                  {report.accuracy_history.length > 1 && (
                    <div className="mt-4">
                      <p className="text-xs font-medium text-muted-foreground mb-2">Accuracy over time</p>
                      <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={report.accuracy_history.map((acc, i) => ({ step: i + 1, accuracy: acc }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={10} />
                            <YAxis stroke="var(--muted-foreground)" fontSize={10} domain={[0, 1]} />
                            <Tooltip
                              contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '11px' }}
                              labelStyle={{ color: 'var(--foreground)' }}
                            />
                            <Line type="monotone" dataKey="accuracy" stroke="var(--primary)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                  {report.caption_history.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Recent captions</p>
                      <div className="max-h-24 overflow-y-auto space-y-0.5">
                        {report.caption_history.slice(-6).reverse().map((c, i) => (
                          <p key={i} className="text-[10px] text-muted-foreground/70 leading-relaxed border-l-2 border-primary/20 pl-2">
                            {c}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Image upload & train */}
            <Card>
              <CardHeader><CardTitle className="text-base">Image Training</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Upload an image to train the vision model. It learns features and generates a caption automatically.</p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {uploading ? 'Training…' : 'Upload image'}
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleUploadImage}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Batch training */}
            <Card>
              <CardHeader><CardTitle className="text-base">Batch Training</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Train on multiple images at once. Upload files or specify a server directory path.</p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => batchFileInputRef.current?.click()}
                    disabled={batchUploading || trainStatus?.running}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {batchUploading ? 'Starting…' : 'Upload images'}
                  </Button>
                  <input
                    ref={batchFileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleBatchUpload}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    value={batchDirPath}
                    onChange={e => setBatchDirPath(e.target.value)}
                    placeholder="/path/to/images on server"
                    className="h-8 text-xs flex-1"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleBatchDir}
                    disabled={!batchDirPath.trim() || trainStatus?.running}
                  >
                    {trainStatus?.running ? 'Training…' : 'Train from directory'}
                  </Button>
                </div>
                {trainStatus && trainStatus.running && (
                  <div className="space-y-1 pt-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{trainStatus.completed}/{trainStatus.total} images</span>
                      <span>{trainStatus.progress_pct}%</span>
                    </div>
                    <ProgressBar value={trainStatus.progress_pct} max={100} variant="default" />
                    {trainStatus.current_image && (
                      <p className="text-[10px] text-muted-foreground/60 truncate">
                        Processing: {trainStatus.current_image}
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* VLM Dataset Creation */}
            <Card>
              <CardHeader><CardTitle className="text-base">VLM Dataset</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">Create a VLM training dataset from a directory of images. Auto-generates captions using the vision model.</p>
                <div className="flex items-center gap-2">
                  <Input
                    value={vlmDatasetName}
                    onChange={e => setVlmDatasetName(e.target.value)}
                    placeholder="Dataset name"
                    className="h-8 text-xs flex-1"
                  />
                  <Input
                    value={vlmImageDir}
                    onChange={e => setVlmImageDir(e.target.value)}
                    placeholder="/path/to/images"
                    className="h-8 text-xs flex-1"
                  />
                </div>
                <Button
                  size="sm"
                  className="h-8 text-xs"
                  onClick={handleCreateVLMDataset}
                  disabled={!vlmDatasetName.trim() || !vlmImageDir.trim() || creatingDataset}
                >
                  {creatingDataset ? 'Creating…' : 'Create dataset'}
                </Button>
              </CardContent>
            </Card>

            {/* VLM Inference */}
            <Card>
              <CardHeader><CardTitle className="text-base">VLM Inference</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  Test a loaded VLM model with an image and text prompt.
                  {vlmLoaded ? (
                    <span className="text-success ml-1">VLM is loaded.</span>
                  ) : (
                    <span className="text-muted-foreground ml-1">Train a VLM on the Training page first.</span>
                  )}
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs"
                    onClick={() => vlmImageInputRef.current?.click()}
                    disabled={!vlmLoaded || vlmInferring}
                  >
                    <IconUpload className="h-3.5 w-3.5 mr-1" />
                    {vlmImageBase64 ? 'Change image' : 'Select image'}
                  </Button>
                  <input
                    ref={vlmImageInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleVLMImageSelect}
                  />
                </div>
                {vlmImageBase64 && (
                  <div className="flex items-center gap-2">
                    <img src={vlmImageBase64} alt="Selected" className="h-16 w-16 rounded object-cover border border-border/50" />
                    <span className="text-xs text-muted-foreground">Image selected for inference</span>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Input
                    value={vlmPrompt}
                    onChange={e => setVlmPrompt(e.target.value)}
                    placeholder="Describe this image in detail..."
                    className="h-8 text-xs flex-1"
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleVLMInfer}
                    disabled={!vlmLoaded || !vlmImageBase64 || vlmInferring}
                  >
                    {vlmInferring ? 'Generating…' : 'Generate'}
                  </Button>
                </div>
                {vlmOutput && (
                  <div className="rounded-md border border-border/40 bg-muted/30 p-3 text-xs leading-relaxed max-h-48 overflow-y-auto">
                    {vlmOutput}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Image generation */}
            <Card>
              <CardHeader><CardTitle className="text-base">Image Generation</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <Input
                    value={genPrompt}
                    onChange={e => setGenPrompt(e.target.value)}
                    placeholder="A cat in a spacesuit…"
                    className="h-8 text-xs flex-1"
                    onKeyDown={e => { if (e.key === 'Enter') handleGenerateImage() }}
                  />
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={handleGenerateImage}
                    disabled={generating || !genPrompt.trim()}
                  >
                    {generating ? 'Generating…' : 'Generate'}
                  </Button>
                </div>
                {generatedImage && (
                  <div className="rounded-lg border border-border/50 overflow-hidden">
                    <img src={generatedImage} alt="Generated" className="w-full h-auto max-h-64 object-contain bg-muted/20" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Audio */}
            <Card>
              <CardHeader><CardTitle className="text-base">Audio</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Speech-to-text</p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs"
                      onClick={() => audioInputRef.current?.click()}
                      disabled={transcribing}
                    >
                      <IconUpload className="h-3.5 w-3.5 mr-1" />
                      {transcribing ? 'Transcribing…' : 'Upload audio'}
                    </Button>
                    <input
                      ref={audioInputRef}
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={handleTranscribe}
                    />
                  </div>
                  {transcript && (
                    <div className="mt-2 p-2 rounded bg-muted/30 border border-border/40 text-xs text-muted-foreground">
                      {transcript}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Text-to-speech</p>
                  <div className="flex items-center gap-2">
                    <Input
                      value={synthText}
                      onChange={e => setSynthText(e.target.value)}
                      placeholder="Text to speak…"
                      className="h-8 text-xs flex-1"
                      onKeyDown={e => { if (e.key === 'Enter') handleSynthesize() }}
                    />
                    <Button
                      size="sm"
                      className="h-8 text-xs shrink-0"
                      onClick={handleSynthesize}
                      disabled={synthesizing || !synthText.trim()}
                    >
                      {synthesizing ? 'Synthesizing…' : 'Speak'}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
