'use client'

import { useState } from 'react'
import { Button, Dialog, DialogContent, DialogHeader, DialogTitle, Input, Textarea } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { downloadMarkdown } from '@/lib/download-utils'
import { FileText, Copy, Download, Sparkles } from 'lucide-react'

interface ModelCardDialogProps {
  open: boolean
  onClose: () => void
  initialName?: string
  initialModel?: string
  initialDataset?: string
}

export function ModelCardDialog({ open, onClose, initialName = '', initialModel = '', initialDataset = '' }: ModelCardDialogProps) {
  const [name, setName] = useState(initialName)
  const [baseModel, setBaseModel] = useState(initialModel || 'gpt2')
  const [description, setDescription] = useState('')
  const [dataset, setDataset] = useState(initialDataset)
  const [epochs, setEpochs] = useState(0)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [batchSize, setBatchSize] = useState(8)
  const [finalLoss, setFinalLoss] = useState(0)
  const [qualityScore, setQualityScore] = useState(0)
  const [markdown, setMarkdown] = useState('')
  const [generating, setGenerating] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleGenerate = async () => {
    if (!name) return
    setGenerating(true)
    try {
      const result = await settingsController.generateModelCard(name, {
        base_model: baseModel,
        description,
        dataset,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        final_loss: finalLoss,
        quality_score: qualityScore,
      })
      setMarkdown(result.markdown)
    } catch (err) {
      console.error('Failed to generate model card:', err)
    } finally {
      setGenerating(false)
    }
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(markdown)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Dialog open={open} onOpenChange={o => { if (!o) onClose() }}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4" /> Generate Model Card
          </DialogTitle>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className="text-xs font-medium">Model Name *</label>
            <Input value={name} onChange={e => setName(e.target.value)} placeholder="my-fine-tuned-model" aria-label="Model name" />
          </div>
          <div>
            <label className="text-xs font-medium">Base Model</label>
            <Input value={baseModel} onChange={e => setBaseModel(e.target.value)} aria-label="Base model" />
          </div>
          <div>
            <label className="text-xs font-medium">Dataset</label>
            <Input value={dataset} onChange={e => setDataset(e.target.value)} aria-label="Dataset" />
          </div>
          <div className="col-span-2">
            <label className="text-xs font-medium">Description</label>
            <Textarea value={description} onChange={e => setDescription(e.target.value)} rows={2} aria-label="Description" />
          </div>
          <div>
            <label className="text-xs font-medium">Epochs</label>
            <Input type="number" value={epochs} onChange={e => setEpochs(Number(e.target.value))} aria-label="Epochs" />
          </div>
          <div>
            <label className="text-xs font-medium">Learning Rate</label>
            <Input type="number" step="0.00001" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} aria-label="Learning rate" />
          </div>
          <div>
            <label className="text-xs font-medium">Batch Size</label>
            <Input type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} aria-label="Batch size" />
          </div>
          <div>
            <label className="text-xs font-medium">Final Loss</label>
            <Input type="number" step="0.01" value={finalLoss} onChange={e => setFinalLoss(Number(e.target.value))} aria-label="Final loss" />
          </div>
          <div>
            <label className="text-xs font-medium">Quality (0-1)</label>
            <Input type="number" step="0.01" min="0" max="1" value={qualityScore} onChange={e => setQualityScore(Number(e.target.value))} aria-label="Quality score" />
          </div>
        </div>
        <Button onClick={handleGenerate} disabled={!name || generating} className="w-full">
          {generating ? 'Generating...' : 'Generate Model Card'}
        </Button>
        {markdown ? (
          <div className="space-y-2">
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={handleCopy}>
                <Copy className="h-4 w-4 mr-1" />{copied ? 'Copied!' : 'Copy'}
              </Button>
              <Button variant="outline" size="sm" onClick={() => downloadMarkdown(markdown, `${name || 'model'}_card.md`)}>
                <Download className="h-4 w-4 mr-1" /> Download .md
              </Button>
            </div>
            <pre className="whitespace-pre-wrap text-xs bg-muted p-3 rounded-lg max-h-[300px] overflow-y-auto font-mono">{markdown}</pre>
          </div>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Fill in the form and click Generate.</p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
