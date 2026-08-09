'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Textarea } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet, apiPost } from '@/lib/http-client'
import { PUBLIC_API_URL } from '@/lib/config'
import { ImageGalleryInsightsCard } from '@/components/images/ImageGalleryInsightsCard'
import { useToastStore } from '@/lib/toast-store'

interface GalleryImage {
  id: string
  path: string
  created: number
}

interface Style {
  key: string
  name: string
}

export default function ImagesPage() {
  const [gallery, setGallery] = useState<GalleryImage[]>([])
  const [styles, setStyles] = useState<Style[]>([])
  const [loading, setLoading] = useState(true)
  const [prompt, setPrompt] = useState('')
  const [selectedStyle, setSelectedStyle] = useState('realistic')
  const [generating, setGenerating] = useState(false)
  const [lastGenerated, setLastGenerated] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchData = async () => {
    try {
      const [galleryRes, stylesRes] = await Promise.all([
        apiGet<{ data?: { images?: GalleryImage[] } }>('/images/gallery').catch(() => null),
        apiGet<{ data?: { styles?: Array<[string, string]> } }>('/images/styles').catch(() => null),
      ])
      setGallery(galleryRes?.data?.images ?? [])
      setStyles((stylesRes?.data?.styles ?? []).map((s: [string, string]) => ({ key: s[0], name: s[1] })))
    } catch {
      addToast('Failed to load image data', 'error')
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
      const data = await apiPost<{ detail?: string; image?: string }>('/images/generate', { prompt, style: selectedStyle })
      if (data.detail) {
        setGenError(data.detail)
        return
      }
      setLastGenerated(data.image ?? null)
      await fetchData()
    } catch (err) {
      setGenError(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Images" subtitle="AI image generation" />} />
        <div className="space-y-4">
          <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Images" subtitle={`${gallery.length} images generated`} />} />
      <div className="space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Generate</CardTitle>
            <Button size="sm" variant="ghost" onClick={fetchData}>
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
                  onClick={() => setSelectedStyle(s.key)}
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
              <p className="text-sm text-muted-foreground">No images generated yet.</p>
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
      </div>
    </div>
  )
}
