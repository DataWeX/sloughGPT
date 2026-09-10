'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { imagesController, type GalleryImage, type ImageStyle } from '@/lib/images-controller'
import { PUBLIC_API_URL } from '@/lib/config'
import { ImageGalleryInsightsCard } from '@/components/images/ImageGalleryInsightsCard'
import { ImageUploaderCard } from '@/components/images/ImageUploaderCard'
import { ImageComparisonCard } from '@/components/images/ImageComparisonCard'
import { ImageHistoryCard, recordImageGeneration } from '@/components/images/ImageHistoryCard'
import { useToastStore } from '@/lib/toast-store'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'

interface Style {
  key: string
  name: string
}

export default function ImagesPage() {
  const router = useRouter()
  const [gallery, setGallery] = useState<GalleryImage[]>([])
  const [styles, setStyles] = useState<Style[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [selectedStyle, setSelectedStyle] = useState<ImageStyle>('realistic')
  const [generating, setGenerating] = useState(false)
  const [lastGenerated, setLastGenerated] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchData = async () => {
    try {
      setLoadError(null)
      const [galleryRes, stylesRes] = await Promise.all([
        imagesController.gallery().catch(() => null),
        imagesController.styles().catch(() => null),
      ])
      setGallery(galleryRes?.images ?? [])
      setStyles((stylesRes?.styles ?? []).map((s: [string, string]) => ({ key: s[0], name: s[1] })))
      if (!galleryRes && !stylesRes) setLoadError('Could not load image data. Please try again.')
    } catch {
      setLoadError('Could not load image data')
    } finally {
      setLoading(false)
    }
  }

  useRefreshShortcut(fetchData)

  useEffect(() => { fetchData() }, [])

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setGenerating(true)
    setGenError(null)
    setLastGenerated(null)
    try {
      const data = await imagesController.generate(prompt, selectedStyle)
      setLastGenerated(data.image ?? null)
      await fetchData()
    } catch (err) {
      setGenError(err instanceof Error ? err.message : 'Could not generation')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <PageContainer
      title="Images"
      subtitle={`${gallery.length} images generated`}
      loading={loading}
      error={loadError}
      onRetry={fetchData}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Generate</CardTitle>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={fetchData} aria-label="Refresh">
            <IconRefresh className="h-3 w-3" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <Textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Describe the image you want to generate..."
            rows={2}
            className="text-[11px] rounded-lg border-border/40"
          />
          <div className="flex flex-wrap gap-1">
            {styles.map(s => (
              <button
                key={s.key}
                type="button"
                onClick={() => setSelectedStyle(s.key as ImageStyle)}
                className={cn('px-2.5 py-1 text-[10px] font-medium rounded-full transition-colors', selectedStyle === s.key ? 'bg-primary/15 text-primary border border-primary/30' : 'bg-muted/30 text-muted-foreground/60 border border-border/40 hover:bg-muted/50')}
              >
                {s.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" className="h-7 text-[11px]" onClick={handleGenerate} disabled={generating || !prompt.trim()}>
              {generating ? 'Generating...' : 'Generate'}
            </Button>
            {lastGenerated && (
              <span className="text-[10px] text-success">Generated</span>
            )}
          </div>
          {genError && <StatusBanner variant="error" message={genError} dismissible={false} />}
        </CardContent>
      </Card>

      {lastGenerated && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Last Generated</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <img src={lastGenerated} alt="Generated" className="w-full max-w-md rounded-lg border border-border/40" />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Gallery ({gallery.length})</CardTitle>
        </CardHeader>
        <CardContent className="px-2.5 pb-2.5">
          {gallery.length === 0 ? (
            <div className="text-center py-6 text-[10px] text-muted-foreground/60 space-y-1.5">
              <div>No images generated yet.</div>
              <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={() => router.push('/chat')}>
                Open Chat
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
              {gallery.map(img => (
                <div key={img.id} className="rounded-lg border border-border/40 overflow-hidden hover:border-border/60 transition-colors">
                  <img
                    src={`${PUBLIC_API_URL}${img.path}`}
                    alt={img.id}
                    className="w-full aspect-square object-cover"
                    loading="lazy"
                  />
                  <div className="px-2 py-1 text-[10px] text-muted-foreground/60 truncate font-mono tabular-nums">
                    {new Date(img.created * 1000).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {gallery.length > 0 && <ImageGalleryInsightsCard gallery={gallery} styles={styles} />}

      <ImageUploaderCard onUpload={(img) => addToast(`Uploaded ${img.name}`, 'success')} />

      <ImageComparisonCard
        images={gallery.slice(0, 10).map(g => ({
          id: g.id,
          label: g.path.split('/').pop() ?? g.id,
          src: `${PUBLIC_API_URL}/static/${g.path}`,
        }))}
      />

      <ImageHistoryCard onReUse={(p, s) => {
        setPrompt(p)
        setSelectedStyle(s as ImageStyle)
      }} />
    </PageContainer>
  )
}
