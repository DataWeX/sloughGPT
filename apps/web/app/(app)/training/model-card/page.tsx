'use client'

export const dynamic = 'force-dynamic'

import { useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Textarea } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { FileText, Copy, Download, Sparkles } from 'lucide-react'

export default function ModelCardPage() {
  const [name, setName] = useState('')
  const [baseModel, setBaseModel] = useState('gpt2')
  const [description, setDescription] = useState('')
  const [dataset, setDataset] = useState('')
  const [datasetSize, setDatasetSize] = useState(0)
  const [epochs, setEpochs] = useState(0)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [batchSize, setBatchSize] = useState(8)
  const [finalLoss, setFinalLoss] = useState(0)
  const [perplexity, setPerplexity] = useState(0)
  const [qualityScore, setQualityScore] = useState(0)
  const [trainingTime, setTrainingTime] = useState(0)
  const [trainingMethod, setTrainingMethod] = useState('finetune')

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
        dataset_size: datasetSize,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        final_loss: finalLoss,
        perplexity,
        quality_score: qualityScore,
        training_time_s: trainingTime,
        training_method: trainingMethod,
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

  const handleDownload = () => {
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name || 'model'}_card.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <PageContainer title="Model Card Generator">
      <AppRouteHeader left={<AppRouteHeaderLead title="Model Card Generator" />} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Model Details
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Model Name *</label>
              <Input value={name} onChange={e => setName(e.target.value)} placeholder="my-fine-tuned-model" />
            </div>
            <div>
              <label className="text-sm font-medium">Base Model</label>
              <Input value={baseModel} onChange={e => setBaseModel(e.target.value)} placeholder="gpt2" />
            </div>
            <div>
              <label className="text-sm font-medium">Description</label>
              <Textarea value={description} onChange={e => setDescription(e.target.value)} placeholder="A fine-tuned model for..." rows={3} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Dataset</label>
                <Input value={dataset} onChange={e => setDataset(e.target.value)} placeholder="dataset-name" />
              </div>
              <div>
                <label className="text-sm font-medium">Dataset Size</label>
                <Input type="number" value={datasetSize} onChange={e => setDatasetSize(Number(e.target.value))} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Epochs</label>
                <Input type="number" value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
              </div>
              <div>
                <label className="text-sm font-medium">Learning Rate</label>
                <Input type="number" step="0.00001" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Batch Size</label>
                <Input type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} />
              </div>
              <div>
                <label className="text-sm font-medium">Training Method</label>
                <Input value={trainingMethod} onChange={e => setTrainingMethod(e.target.value)} placeholder="finetune" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Final Loss</label>
                <Input type="number" step="0.01" value={finalLoss} onChange={e => setFinalLoss(Number(e.target.value))} />
              </div>
              <div>
                <label className="text-sm font-medium">Perplexity</label>
                <Input type="number" step="0.01" value={perplexity} onChange={e => setPerplexity(Number(e.target.value))} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Quality Score (0-1)</label>
                <Input type="number" step="0.01" min="0" max="1" value={qualityScore} onChange={e => setQualityScore(Number(e.target.value))} />
              </div>
              <div>
                <label className="text-sm font-medium">Training Time (s)</label>
                <Input type="number" value={trainingTime} onChange={e => setTrainingTime(Number(e.target.value))} />
              </div>
            </div>
            <Button onClick={handleGenerate} disabled={!name || generating} className="w-full">
              {generating ? 'Generating...' : 'Generate Model Card'}
            </Button>
          </CardContent>
        </Card>

        {/* Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Preview
            </CardTitle>
          </CardHeader>
          <CardContent>
            {markdown ? (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    <Copy className="h-4 w-4 mr-1" />
                    {copied ? 'Copied!' : 'Copy'}
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleDownload}>
                    <Download className="h-4 w-4 mr-1" />
                    Download .md
                  </Button>
                </div>
                <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded-lg max-h-[600px] overflow-y-auto font-mono">
                  {markdown}
                </pre>
              </div>
            ) : (
              <div className="text-center py-16 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Fill in the form and click Generate to create a model card.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
