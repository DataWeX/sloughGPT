'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { IconRefresh, IconUpload } from '@/components/ui'
import { StatCard, KpiGrid } from '@/components/strui'
import { useToastStore } from '@/lib/toast-store'
import { visualController } from '@/lib/controllers'
import { cn } from '@/lib/cn'

export default function VisualPage() {
  const addToast = useToastStore(s => s.addToast)

  // ── State ──────────────────────────────────────────────────────
  const [visualStatus, setVisualStatus] = useState<{ loaded: boolean; model?: string; vision_encoder?: string; llm?: string } | null>(null)
  const [trainStatus, setTrainStatus] = useState<any>(null)
  const [dpoStatus, setDpoStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  // ── Video Training ──────────────────────────────────────────────
  const [videoDataPath, setVideoDataPath] = useState('')
  const [videoEpochs, setVideoEpochs] = useState(5)
  const [videoBatchSize, setVideoBatchSize] = useState(2)
  const [videoTrainRunning, setVideoTrainRunning] = useState(false)
  const [videoTrainStatus, setVideoTrainStatus] = useState<{
    status: string; current_epoch: number; current_step: number
    total_steps: number; current_loss: number | null; result: any; error: string | null
  } | null>(null)

  // ── Video Inference ──────────────────────────────────────────────
  const [videoInferPath, setVideoInferPath] = useState('')
  const [videoInferResult, setVideoInferResult] = useState<string | null>(null)
  const [videoInferRunning, setVideoInferRunning] = useState(false)

  // ── Inference ──────────────────────────────────────────────────
  const [inferImage, setInferImage] = useState<string | null>(null)
  const [inferPrompt, setInferPrompt] = useState('Describe this image in detail.')
  const [inferResult, setInferResult] = useState<string | null>(null)
  const [inferRunning, setInferRunning] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Dataset ────────────────────────────────────────────────────
  const [datasetName, setDatasetName] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  const [creatingDataset, setCreatingDataset] = useState(false)

  // ── Training ───────────────────────────────────────────────────
  const [trainingDataPath, setTrainingDataPath] = useState('')
  const [stage1Epochs, setStage1Epochs] = useState(1)
  const [stage2Epochs, setStage2Epochs] = useState(2)
  const [startingTrain, setStartingTrain] = useState(false)

  // ── Load ───────────────────────────────────────────────────────
  const [loadModelDir, setLoadModelDir] = useState('models/visual-finetuned')
  const [loadingModel, setLoadingModel] = useState(false)

  // ── Checkpoints ────────────────────────────────────────────────
  const [checkpoints, setCheckpoints] = useState<Array<{
    name: string; path: string; size_mb: number; final_loss: number | null;
    total_steps: number; vision_encoder?: string; llm?: string
  }>>([])
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)

  // ── PDF Analysis ────────────────────────────────────────────────
  const [pdfPath, setPdfPath] = useState('')
  const [pdfQuestion, setPdfQuestion] = useState('Summarize this document.')
  const [pdfAnalyzing, setPdfAnalyzing] = useState(false)
  const [pdfResult, setPdfResult] = useState<string | null>(null)
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [pdfPerPage, setPdfPerPage] = useState(false)
  const pdfFileInputRef = useRef<HTMLInputElement>(null)

  // ── Fetch all status ───────────────────────────────────────────
  const fetchAll = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true)
    try {
      const [vs, ts, ds] = await Promise.all([
        visualController.getVisualStatus().catch(() => null),
        visualController.getVisualTrainStatus().catch(() => null),
        visualController.getDPOStatus().catch(() => null),
      ])
      setVisualStatus(vs)
      setTrainStatus(ts)
      setDpoStatus(ds)
    } catch {}
    setLoading(false)
    setRefreshing(false)
  }, [])

  const fetchCheckpoints = useCallback(async () => {
    setLoadingCheckpoints(true)
    try {
      const result = await visualController.listCheckpoints()
      setCheckpoints(result.checkpoints)
    } catch {} finally {
      setLoadingCheckpoints(false)
    }
  }, [])

  useEffect(() => { fetchAll(); fetchCheckpoints() }, [fetchAll, fetchCheckpoints])

  // ── Handlers ───────────────────────────────────────────────────

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setInferImage(reader.result as string)
    reader.readAsDataURL(file)
  }

  const handleInfer = async () => {
    if (!inferImage) return
    setInferRunning(true)
    setInferResult(null)
    try {
      const result = await visualController.visualInference(inferImage, inferPrompt)
      setInferResult(result.text)
      addToast(`Visual inference: ${result.tokens_generated} tokens in ${result.elapsed_ms}ms`, 'success')
    } catch (err: any) {
      addToast(`Inference failed: ${err.message}`, 'error')
    } finally {
      setInferRunning(false)
    }
  }

  const handleCreateDataset = async () => {
    if (!datasetName || !datasetDir) return
    setCreatingDataset(true)
    try {
      const result = await visualController.createVisualDataset(datasetName, datasetDir)
      addToast(`Dataset created: ${result.entries} entries from ${datasetDir}`, 'success')
      setDatasetName('')
      setDatasetDir('')
      fetchAll()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    } finally {
      setCreatingDataset(false)
    }
  }

  const handleStartTrain = async () => {
    if (!trainingDataPath) return
    setStartingTrain(true)
    try {
      const result = await visualController.startVisualTrain({
        data_path: trainingDataPath,
        stage1_epochs: stage1Epochs,
        stage2_epochs: stage2Epochs,
      })
      addToast(`Training started: ${result.job_id}`, 'success')
      fetchAll()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    } finally {
      setStartingTrain(false)
    }
  }

  const handleLoadModel = async () => {
    setLoadingModel(true)
    try {
      const result = await visualController.loadVisualModel(loadModelDir)
      addToast(result.message, 'success')
      fetchAll()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    } finally {
      setLoadingModel(false)
    }
  }

  const handleLoadCheckpoint = async (name: string) => {
    try {
      await visualController.loadCheckpoint(name)
      addToast(`Checkpoint '${name}' loaded`, 'success')
      fetchAll(); fetchCheckpoints()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    }
  }

  const handleDeleteCheckpoint = async (name: string) => {
    try {
      await visualController.deleteCheckpoint(name)
      addToast(`Checkpoint '${name}' deleted`, 'success')
      fetchCheckpoints()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    }
  }

  const handleImportFromCheckpoint = (path: string) => {
    setLoadModelDir(path)
    addToast('Model directory set from checkpoint', 'info')
  }

  // ── Video Handlers ────────────────────────────────────────────

  const handleStartVideoTrain = async () => {
    if (!videoDataPath) return
    setVideoTrainRunning(true)
    try {
      const result = await visualController.startVideoTrain({
        data_path: videoDataPath,
        epochs: videoEpochs,
        batch_size: videoBatchSize,
      })
      addToast(`Video training started: ${result.job_id}`, 'success')

      // Poll for status
      const poll = setInterval(async () => {
        try {
          const st = await visualController.getVideoTrainStatus()
          setVideoTrainStatus(st)
          if (st.status === 'completed' || st.status === 'error') {
            clearInterval(poll)
            setVideoTrainRunning(false)
            if (st.status === 'completed') {
              addToast('Video training complete!', 'success')
            } else if (st.error) {
              addToast(`Video training failed: ${st.error}`, 'error')
            }
          }
        } catch { clearInterval(poll); setVideoTrainRunning(false) }
      }, 3000)
    } catch (err: any) {
      addToast(`Failed to start video training: ${err.message}`, 'error')
      setVideoTrainRunning(false)
    }
  }

  const handleVideoInfer = async () => {
    if (!videoInferPath) return
    setVideoInferRunning(true)
    setVideoInferResult(null)
    try {
      const result = await visualController.videoInference({
        video_path: videoInferPath,
      })
      setVideoInferResult(result.text)
      addToast(`Video inference: ${result.checkpoint} (${result.elapsed_ms}ms)`, 'info')
    } catch (err: any) {
      addToast(`Video inference failed: ${err.message}`, 'error')
    } finally {
      setVideoInferRunning(false)
    }
  }

  // ── PDF Handlers ────────────────────────────────────────────────

  const handlePDFUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) setPdfFile(file)
  }

  const handleAnalyzePDF = async () => {
    if (!pdfPath.trim() && !pdfFile) return
    setPdfAnalyzing(true)
    setPdfResult(null)
    try {
      if (pdfFile) {
        const result = await visualController.analyzePDFUpload(pdfFile, pdfQuestion, pdfPerPage)
        setPdfResult(result.analysis || (result.pages || []).map(p => p.text).join('\n\n---\n\n'))
      } else {
        const result = await visualController.analyzePDF({
          pdf_path: pdfPath.trim(),
          question: pdfQuestion,
          per_page: pdfPerPage,
        })
        setPdfResult(result.analysis || (result.pages || []).map(p => p.text).join('\n\n---\n\n'))
      }
      addToast('PDF analysis complete', 'success')
    } catch (err: any) {
      addToast(`PDF analysis failed: ${err.message}`, 'error')
    } finally {
      setPdfAnalyzing(false)
    }
  }

  // ── Render ─────────────────────────────────────────────────────

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={
          <AppRouteHeaderLead
            title="Visual AI"
            subtitle="Load, train, and run visual inference on images"
          />
        }
        right={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fetchAll(true)}
            disabled={refreshing}
          >
            <IconRefresh className={cn("h-4 w-4", refreshing && "animate-spin")} />
          </Button>
        }
      />

      <div className="space-y-4">
        {/* Status */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Status</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-20 animate-pulse rounded bg-muted" />
            ) : (
              <KpiGrid columns={4}>
                <StatCard
                  label="Visual AI Loaded"
                  value={visualStatus?.loaded ? 'Yes' : 'No'}
                  icon={<span className={cn("w-2 h-2 rounded-full", visualStatus?.loaded ? 'bg-success' : 'bg-muted-foreground/40')} />}
                />
                <StatCard label="Vision Encoder" value={visualStatus?.vision_encoder || '—'} />
                <StatCard label="LLM" value={visualStatus?.llm || '—'} />
                <StatCard
                  label="DPO Status"
                  value={dpoStatus?.status || 'idle'}
                  icon={<span className={cn("w-2 h-2 rounded-full",
                    dpoStatus?.status === 'running' ? 'bg-warning animate-pulse' :
                    dpoStatus?.status === 'completed' ? 'bg-success' : 'bg-muted-foreground/40'
                  )} />}
                />
              </KpiGrid>
            )}
          </CardContent>
        </Card>

        {/* Inference */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Image Inference</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-3">
              <div className="flex-1">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageUpload}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full"
                >
                  <IconUpload className="h-4 w-4 mr-2" />
                  {inferImage ? 'Change Image' : 'Upload Image'}
                </Button>
                {inferImage && (
                  <img src={inferImage} alt="Upload" className="mt-2 h-32 w-32 object-cover rounded-lg border" />
                )}
              </div>
              <div className="flex-1 space-y-2">
                <Input
                  value={inferPrompt}
                  onChange={(e) => setInferPrompt(e.target.value)}
                  placeholder="What to ask about this image?"
                  className="text-sm"
                />
                <Button
                  size="sm"
                  onClick={handleInfer}
                  disabled={!inferImage || inferRunning}
                  className="w-full"
                >
                  {inferRunning ? 'Analyzing...' : 'Run Inference'}
                </Button>
              </div>
            </div>
            {inferResult && (
              <div className="rounded-lg bg-muted/50 p-3 text-sm font-mono leading-relaxed">
                {inferResult}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dataset Creation */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Create Visual Dataset</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="Dataset name"
              className="text-sm"
            />
            <Input
              value={datasetDir}
              onChange={(e) => setDatasetDir(e.target.value)}
              placeholder="Path to image directory on server"
              className="text-sm"
            />
            <Button
              size="sm"
              onClick={handleCreateDataset}
              disabled={!datasetName || !datasetDir || creatingDataset}
            >
              {creatingDataset ? 'Creating...' : 'Create Dataset'}
            </Button>
          </CardContent>
        </Card>

        {/* Training */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Train Visual AI</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={trainingDataPath}
              onChange={(e) => setTrainingDataPath(e.target.value)}
              placeholder="Training data path (JSONL)"
              className="text-sm"
            />
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground">Stage 1 Epochs</label>
                <Input
                  type="number"
                  value={stage1Epochs}
                  onChange={(e) => setStage1Epochs(parseInt(e.target.value) || 1)}
                  className="text-sm"
                  min={1}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Stage 2 Epochs</label>
                <Input
                  type="number"
                  value={stage2Epochs}
                  onChange={(e) => setStage2Epochs(parseInt(e.target.value) || 2)}
                  className="text-sm"
                  min={1}
                />
              </div>
            </div>
            <Button
              size="sm"
              onClick={handleStartTrain}
              disabled={!trainingDataPath || startingTrain || trainStatus?.status === 'running'}
            >
              {startingTrain ? 'Starting...' : trainStatus?.status === 'running' ? 'Training...' : 'Start Training'}
            </Button>
            {trainStatus?.status === 'running' && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  Stage: {trainStatus.current_stage} | Step {trainStatus.current_step}/{trainStatus.total_steps}
                </p>
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-primary transition-all" style={{ width: `${(trainStatus.progress || 0) * 100}%` }} />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Video Training */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Video Training</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={videoDataPath}
              onChange={(e) => setVideoDataPath(e.target.value)}
              placeholder="Video JSONL data path"
              className="text-sm"
            />
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground">Epochs</label>
                <Input
                  type="number"
                  value={videoEpochs}
                  onChange={(e) => setVideoEpochs(parseInt(e.target.value) || 5)}
                  className="text-sm"
                  min={1}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Batch Size</label>
                <Input
                  type="number"
                  value={videoBatchSize}
                  onChange={(e) => setVideoBatchSize(parseInt(e.target.value) || 2)}
                  className="text-sm"
                  min={1}
                />
              </div>
            </div>
            <Button
              size="sm"
              onClick={handleStartVideoTrain}
              disabled={!videoDataPath || videoTrainRunning || videoTrainStatus?.status === 'running'}
            >
              {videoTrainRunning ? 'Starting...' : videoTrainStatus?.status === 'running' ? 'Training...' : 'Start Video Training'}
            </Button>
            {videoTrainStatus?.status === 'running' && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  Epoch {videoTrainStatus.current_epoch} | Step {videoTrainStatus.current_step}/{videoTrainStatus.total_steps}
                  {videoTrainStatus.current_loss != null && ` | loss: ${videoTrainStatus.current_loss.toFixed(4)}`}
                </p>
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-primary transition-all" style={{ width: `${videoTrainStatus.total_steps > 0 ? (videoTrainStatus.current_step / videoTrainStatus.total_steps) * 100 : 0}%` }} />
                </div>
              </div>
            )}
            {videoTrainStatus?.status === 'completed' && (
              <p className="text-xs text-success">Training complete!</p>
            )}
          </CardContent>
        </Card>

        {/* Video Inference */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Video Inference</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={videoInferPath}
              onChange={(e) => setVideoInferPath(e.target.value)}
              placeholder="Server path to video file"
              className="text-sm"
            />
            <Button
              size="sm"
              onClick={handleVideoInfer}
              disabled={!videoInferPath || videoInferRunning}
            >
              {videoInferRunning ? 'Generating...' : 'Generate Caption'}
            </Button>
            {videoInferResult && (
              <div className="rounded-lg bg-muted/50 p-3 text-sm font-mono leading-relaxed">
                {videoInferResult}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Load Model */}
        <Card>
          <CardHeader>
              <CardTitle className="text-base">Load Visual Model</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Input
              value={loadModelDir}
              onChange={(e) => setLoadModelDir(e.target.value)}
              placeholder="Model directory path"
              className="text-sm"
            />
            <Button
              size="sm"
              onClick={handleLoadModel}
              disabled={loadingModel}
            >
              {loadingModel ? 'Loading...' : 'Load Model'}
            </Button>
          </CardContent>
        </Card>

        {/* PDF Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">PDF Analysis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Analyze PDF documents using the visual model. Provide a server path or upload a file.
            </p>

            {/* Source toggle: path vs upload */}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={pdfFile ? 'outline' : 'default'}
                onClick={() => { setPdfFile(null); setPdfPath('') }}
              >
                Server Path
              </Button>
              <Button
                size="sm"
                variant={pdfFile ? 'default' : 'outline'}
                onClick={() => pdfFileInputRef.current?.click()}
              >
                Upload File
              </Button>
              <input
                ref={pdfFileInputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handlePDFUpload}
              />
            </div>

            {pdfFile ? (
              <p className="text-xs text-muted-foreground truncate">Selected: {pdfFile.name}</p>
            ) : (
              <Input
                value={pdfPath}
                onChange={(e) => setPdfPath(e.target.value)}
                placeholder="Server path to PDF — e.g. /home/user/doc.pdf"
                className="text-sm"
              />
            )}

            <Input
              value={pdfQuestion}
              onChange={(e) => setPdfQuestion(e.target.value)}
              placeholder="What to ask about this document?"
              className="text-sm"
            />

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pdfPerPage}
                  onChange={(e) => setPdfPerPage(e.target.checked)}
                  className="rounded"
                />
                <span className="text-xs text-muted-foreground">Analyze per page</span>
              </label>
              <Button
                size="sm"
                onClick={handleAnalyzePDF}
                disabled={pdfAnalyzing || (!pdfPath.trim() && !pdfFile)}
                className="ml-auto"
              >
                {pdfAnalyzing ? 'Analyzing...' : 'Analyze'}
              </Button>
            </div>

            {pdfResult && (
              <div className="rounded-lg bg-muted/50 p-3 text-sm leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">
                {pdfResult}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Checkpoints */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Saved Checkpoints</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {loadingCheckpoints ? (
              <div className="h-16 animate-pulse rounded bg-muted" />
            ) : checkpoints.length === 0 ? (
              <p className="text-xs text-muted-foreground">No visual checkpoints found on disk.</p>
            ) : (
              checkpoints.map((ck) => (
                <div key={ck.name} className="flex items-center justify-between rounded-lg border p-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate">{ck.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {ck.size_mb.toFixed(2)} MB
                      {ck.final_loss != null && ` · loss: ${ck.final_loss.toFixed(4)}`}
                      {ck.total_steps > 0 && ` · ${ck.total_steps} steps`}
                      {ck.llm && ` · ${ck.llm}`}
                    </p>
                  </div>
                  <div className="flex gap-1 ml-3 shrink-0">
                    <Button size="sm" variant="ghost" onClick={() => handleImportFromCheckpoint(ck.path)}>
                      Use Path
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleLoadCheckpoint(ck.name)}>
                      Load
                    </Button>
                    <Button size="sm" variant="destructive" onClick={() => handleDeleteCheckpoint(ck.name)}>
                      Delete
                    </Button>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
