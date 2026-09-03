'use client'

import { useState, useRef, useMemo } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { Spinner } from '@sloughgpt/strui'
import { datasetController } from '@/lib/dataset-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'

const KAGGLE_PRESETS = [
  { slug: 'heptapod/titanic', title: 'Titanic', desc: 'Classic ML dataset' },
  { slug: 'uciml/iris', title: 'Iris', desc: 'Flower classification' },
  { slug: 'zillow/zecon', title: 'Zillow Housing', desc: 'Home value estimates' },
  { slug: 'dgomonov/new-york-city-airbnb-open-data', title: 'NYC Airbnb', desc: 'Listings & reviews' },
  { slug: 'rounakbanik/pokemon', title: 'Pokemon', desc: 'All Pokemon stats' },
  { slug: 'unsdsn/world-happiness', title: 'Happiness', desc: 'World Happiness Report' },
  { slug: 'rsrishav/youtube-trending-video-dataset', title: 'YouTube Trending', desc: 'Daily trending videos' },
  { slug: 'nelgiriyewithana/global-weather-repository', title: 'World Weather', desc: 'Daily weather data' },
  { slug: 'ashirwadsangwan/imdb-dataset', title: 'IMDb Movies', desc: 'Full IMDb database' },
  { slug: 'saurabhshahane/statsbomb-football-data', title: 'Football Stats', desc: 'StatsBomb data' },
  { slug: 'rsrishav/world-population', title: 'World Population', desc: '2021 population data' },
  { slug: 'markmedhat/supermarket-sales', title: 'Supermarket Sales', desc: 'Sales transaction data' },
]

function shuffleAndPick(arr: typeof KAGGLE_PRESETS, n: number) {
  const shuffled = [...arr].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, n)
}

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImported: () => void
}

export default function QuickImportDialog({ open, onOpenChange, onImported }: Props) {
  const addToast = useToastStore(s => s.addToast)
  const [activeTab, setActiveTab] = useState('local')
  const [name, setName] = useState('')
  const [importing, setImporting] = useState(false)
  const [result, setResult] = useState<{ datasetId: string; message: string } | null>(null)

  const [localPath, setLocalPath] = useState('')
  const [githubUrl, setGithubUrl] = useState('')
  const [hfId, setHfId] = useState('')
  const [url, setUrl] = useState('')
  const [kaggleSlug, setKaggleSlug] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  const randomPresets = useMemo(() => shuffleAndPick(KAGGLE_PRESETS, 4), [open])

  const reset = () => { setName(''); setLocalPath(''); setGithubUrl(''); setHfId(''); setUrl(''); setKaggleSlug(''); setResult(null) }

  const handleImport = async () => {
    const ac = new AbortController()
    abortRef.current = ac
    setImporting(true); setResult(null)
    try {
      const signal = ac.signal
      let res
      if (activeTab === 'local' && localPath.trim()) {
        res = await datasetController.importFromLocal({ path: localPath.trim(), name: name.trim() || 'imported_dataset' }, { signal })
      } else if (activeTab === 'github' && githubUrl.trim()) {
        res = await datasetController.importFromGitHub({ url: githubUrl.trim(), name: name.trim() || 'imported_dataset' }, { signal })
      } else if (activeTab === 'huggingface' && hfId.trim()) {
        res = await datasetController.importFromHuggingFace({ dataset_id: hfId.trim(), name: name.trim() || undefined }, { signal })
      } else if (activeTab === 'url' && url.trim()) {
        res = await datasetController.importFromURL({ url: url.trim(), name: name.trim() || 'imported_dataset' }, { signal })
      } else if (activeTab === 'kaggle' && kaggleSlug.trim()) {
        res = await datasetController.importFromKaggle({ dataset: kaggleSlug.trim(), name: name.trim() || undefined }, { signal })
      } else {
        addToast('Fill in the required field', 'error'); setImporting(false); return
      }
      setResult({ datasetId: res.dataset_id, message: res.message })
      addToast(res.message || 'Imported successfully', 'success')
      onImported()
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      addToast(extractErrorMessage(err, 'Could not import'), 'error')
    } finally {
      abortRef.current = null
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
                <TabsTrigger value="kaggle" className="text-xs">Kaggle</TabsTrigger>
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
              <TabsContent value="kaggle" className="space-y-2">
                <Input placeholder="owner/dataset-name" value={kaggleSlug} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setKaggleSlug(e.target.value)} className="h-8 text-xs" />
                <div className="grid grid-cols-2 gap-1.5">
                  {randomPresets.map((p) => (
                    <button
                      key={p.slug}
                      type="button"
                      className="text-left p-1.5 rounded border border-border/50 hover:border-primary/50 hover:bg-accent/50 transition-colors"
                      onClick={() => setKaggleSlug(p.slug)}
                    >
                      <span className="text-xs font-medium">{p.title}</span>
                      <span className="block text-[10px] text-muted-foreground">{p.desc}</span>
                    </button>
                  ))}
                </div>
              </TabsContent>
              <TabsContent value="url">
                <Input placeholder="https://example.com/data.txt" value={url} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)} className="h-8 text-xs" />
              </TabsContent>
            </Tabs>
            {importing ? (
              <Button className="w-full h-8 text-xs" variant="outline" onClick={() => abortRef.current?.abort()}>
                Cancel Import
              </Button>
            ) : (
              <Button className="w-full h-8 text-xs" onClick={handleImport}>
                Import
              </Button>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
