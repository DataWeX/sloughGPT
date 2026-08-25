'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { imagesController, type GalleryImage, type ImageStyle } from '@/lib/images-controller'
import { PUBLIC_API_URL } from '@/lib/config'
import { ImageGalleryInsightsCard } from '@/components/images/ImageGalleryInsightsCard'
import { useToastStore } from '@/lib/toast-store'

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
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Generate</CardTitle>
          <Button size="sm" variant="ghost" onClick={fetchData} aria-label="Refresh">
            <IconRefresh className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Describe the image you want to generate..."
            rows={2}
          />
          <div className="flex flex-wrap gap-2">
            {styles.map(s => (
              <button
                key={s.key}
                type="button"
                onClick={() => setSelectedStyle(s.key as ImageStyle)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  selectedStyle === s.key
                    ? 'bg-primary/15 text-primary border border-primary/30'
                    : 'bg-muted/50 text-muted-foreground border border-border/60 hover:bg-muted'
                }`}
              >
                {s.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <Button size="sm" onClick={handleGenerate} disabled={generating || !prompt.trim()}>
              {generating ? 'Generating...' : 'Generate'}
            </Button>
            {lastGenerated && (
              <span className="text-xs text-success">Generated</span>
            )}
          </div>
          {genError && <div className="text-xs text-destructive">{genError}</div>}
        </CardContent>
      </Card>

      {lastGenerated && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Last Generated</CardTitle>
          </CardHeader>
          <CardContent>
            <img src={lastGenerated} alt="Generated" className="w-full max-w-md rounded-md border border-border/60" />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Gallery ({gallery.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {gallery.length === 0 ? (
            <div className="text-center py-8 text-sm text-muted-foreground space-y-2">
              <div>No images generated yet.</div>
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/chat')}>
                Open Chat
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {gallery.map(img => (
                <div key={img.id} className="rounded-md border border-border/60 overflow-hidden hover:border-border transition-colors">
                  <img
                    src={`${PUBLIC_API_URL}${img.path}`}
                    alt={img.id}
                    className="w-full aspect-square object-cover"
                    loading="lazy"
                  />
                  <div className="px-2 py-1 text-xs text-muted-foreground truncate">
                    {new Date(img.created * 1000).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {gallery.length > 0 && <ImageGalleryInsightsCard gallery={gallery} styles={styles} />}
    </PageContainer>
  )
}
