'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'
import { datasetController } from '@/lib/dataset-controller'
import { useToastStore } from '@/lib/toast-store'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImported: () => void
}

export default function DatasetInlineImportModal({ open, onOpenChange, onImported }: Props) {
  const addToast = useToastStore(s => s.addToast)
  const [activeTab, setActiveTab] = useState('local')
  const [name, setName] = useState('')
  const [importing, setImporting] = useState(false)
  const [result, setResult] = useState<{ datasetId: string; message: string } | null>(null)

  const [localPath, setLocalPath] = useState('')
  const [githubUrl, setGithubUrl] = useState('')
  const [hfId, setHfId] = useState('')
  const [url, setUrl] = useState('')

  const reset = () => { setName(''); setLocalPath(''); setGithubUrl(''); setHfId(''); setUrl(''); setResult(null) }

  const handleImport = async () => {
    setImporting(true); setResult(null)
    try {
      let res
      if (activeTab === 'local' && localPath.trim()) {
        res = await datasetController.importFromLocal({ path: localPath.trim(), name: name.trim() || 'imported_dataset' })
      } else if (activeTab === 'github' && githubUrl.trim()) {
        res = await datasetController.importFromGitHub({ url: githubUrl.trim(), name: name.trim() || 'imported_dataset' })
      } else if (activeTab === 'huggingface' && hfId.trim()) {
        res = await datasetController.importFromHuggingFace({ dataset_id: hfId.trim(), name: name.trim() || undefined })
      } else if (activeTab === 'url' && url.trim()) {
        res = await datasetController.importFromURL({ url: url.trim(), name: name.trim() || 'imported_dataset' })
      } else {
        addToast('Fill in the required field', 'error'); setImporting(false); return
      }
      setResult({ datasetId: res.dataset_id, message: res.message })
      addToast(res.message || 'Imported successfully', 'success')
      onImported()
    } catch {
      addToast('Import failed', 'error')
    } finally {
      setImporting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { onOpenChange(o); if (!o) reset() }}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Import Dataset</DialogTitle>
        </DialogHeader>
        {result ? (
          <div className="space-y-3">
            <div className="text-sm text-center py-4">
              <p className="font-medium">Imported successfully</p>
              <p className="text-muted-foreground mt-1">{result.message}</p>
            </div>
            <Button className="w-full" onClick={() => { onOpenChange(false); reset() }}>Done</Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Input placeholder="Dataset name (optional)" value={name} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)} className="h-8 text-xs" />
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="h-8">
                <TabsTrigger value="local" className="text-xs">Local Path</TabsTrigger>
                <TabsTrigger value="github" className="text-xs">GitHub</TabsTrigger>
                <TabsTrigger value="huggingface" className="text-xs">HuggingFace</TabsTrigger>
                <TabsTrigger value="url" className="text-xs">URL</TabsTrigger>
              </TabsList>
              <TabsContent value="local">
                <Input placeholder="/path/to/dataset/folder" value={localPath} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setLocalPath(e.target.value)} className="h-8 text-xs" />
              </TabsContent>
              <TabsContent value="github">
                <Input placeholder="https://github.com/user/repo" value={githubUrl} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setGithubUrl(e.target.value)} className="h-8 text-xs" />
              </TabsContent>
              <TabsContent value="huggingface">
                <Input placeholder="username/dataset-name" value={hfId} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setHfId(e.target.value)} className="h-8 text-xs" />
              </TabsContent>
              <TabsContent value="url">
                <Input placeholder="https://example.com/data.txt" value={url} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)} className="h-8 text-xs" />
              </TabsContent>
            </Tabs>
            <Button className="w-full h-8 text-xs" onClick={handleImport} disabled={importing}>
              {importing ? <><Spinner className="h-3.5 w-3.5 mr-1.5" /> Importing...</> : 'Import'}
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
