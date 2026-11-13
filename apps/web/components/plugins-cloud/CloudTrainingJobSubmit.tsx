'use client'

import { useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Input,
} from '@sloughgpt/strui'

interface CloudTrainingJobSubmitProps {
  onSubmit?: (provider: string, datasetId: string) => Promise<void>
  submitting?: boolean
}

export function CloudTrainingJobSubmit({
  onSubmit,
  submitting = false,
}: CloudTrainingJobSubmitProps) {
  const [provider, setProvider] = useState('local')
  const [datasetId, setDatasetId] = useState('')

  const handleSubmit = async () => {
    if (!datasetId || !onSubmit) return
    await onSubmit(provider, datasetId)
    setDatasetId('')
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Submit Training Job</CardTitle>
        <CardDescription>Train a model on cloud infrastructure</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Provider</label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full border border-border rounded-md px-3 py-2 text-sm bg-background"
            >
              <option value="local">Local</option>
              <option value="aws">AWS SageMaker</option>
              <option value="gcp">GCP Vertex AI</option>
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Dataset ID</label>
            <Input
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              placeholder="e.g. my-dataset"
            />
          </div>
        </div>
        <Button onClick={handleSubmit} disabled={submitting || !datasetId}>
          {submitting ? 'Submitting...' : 'Submit Job'}
        </Button>
      </CardContent>
    </Card>
  )
}
