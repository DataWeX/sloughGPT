'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Input,
  Badge,
  Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'

interface CloudJob {
  job_id: string
  provider: string
  status: string
  progress: number
  error?: string
}

interface PluginInfo {
  name: string
  version: string
  description: string
  author: string
  enabled: boolean
}

export default function PluginsCloudPage() {
  const [activeTab, setActiveTab] = useState<'cloud' | 'plugins'>('cloud')
  const [cloudJobs, setCloudJobs] = useState<CloudJob[]>([])
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [provider, setProvider] = useState('local')
  const [datasetId, setDatasetId] = useState('')
  const addToast = useToastStore(s => s.addToast)

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [jobsResp, pluginsResp] = await Promise.all([
        apiGet<{ jobs: CloudJob[] }>('/cloud-training/jobs').catch(() => ({ jobs: [] })),
        apiGet<{ plugins: PluginInfo[] }>('/plugins').catch(() => ({ plugins: [] })),
      ])
      setCloudJobs(jobsResp.jobs || [])
      setPlugins(pluginsResp.plugins || [])
    } catch (err) {
      console.error('Failed to load data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const startJob = async () => {
    if (!datasetId) {
      addToast('Dataset required', 'error')
      return
    }
    setSubmitting(true)
    try {
      await apiPost('/cloud-training/submit', { provider, dataset_id: datasetId })
      addToast('Job submitted', 'success')
      setDatasetId('')
      loadData()
    } catch (err) {
      addToast('Failed to submit', 'error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <PageContainer title="Plugins & Cloud Training">
      <div className="border-b border-border/50 mb-6">
        <div className="flex gap-0">
          {(['cloud', 'plugins'] as const).map(tab => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`relative px-4 py-2.5 text-xs font-medium transition-colors ${
                activeTab === tab ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab === 'cloud' ? 'Cloud Training' : 'Plugins'}
              {activeTab === tab && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full" />
              )}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'cloud' && (
        <div className="space-y-6">
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
              <Button onClick={startJob} disabled={submitting || !datasetId}>
                {submitting ? 'Submitting...' : 'Submit Job'}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Training Jobs</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-20 w-full" />
              ) : cloudJobs.length === 0 ? (
                <p className="text-sm text-muted-foreground">No training jobs yet.</p>
              ) : (
                <div className="space-y-2">
                  {cloudJobs.map(job => (
                    <div key={job.job_id} className="flex items-center justify-between p-2 border border-border/50 rounded">
                      <div>
                        <span className="font-mono text-xs">{job.job_id}</span>
                        <span className="ml-2 text-xs text-muted-foreground">{job.provider}</span>
                      </div>
                      <Badge variant={job.status === 'completed' ? 'default' : job.status === 'failed' ? 'destructive' : 'secondary'}>
                        {job.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'plugins' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Installed Plugins</CardTitle>
              <CardDescription>Manage loaded plugins</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <Skeleton className="h-20 w-full" />
              ) : plugins.length === 0 ? (
                <p className="text-sm text-muted-foreground">No plugins installed. Add plugins to ~/.config/sloughgpt/plugins/</p>
              ) : (
                <div className="space-y-2">
                  {plugins.map(plugin => (
                    <div key={plugin.name} className="flex items-center justify-between p-2 border border-border/50 rounded">
                      <div>
                        <span className="font-medium text-sm">{plugin.name}</span>
                        <span className="ml-2 text-xs text-muted-foreground">v{plugin.version}</span>
                        <span className="ml-2 text-xs text-muted-foreground">by {plugin.author}</span>
                      </div>
                      <Badge variant={plugin.enabled ? 'default' : 'secondary'}>
                        {plugin.enabled ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </PageContainer>
  )
}
