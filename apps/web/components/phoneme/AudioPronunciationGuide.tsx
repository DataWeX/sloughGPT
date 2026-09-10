'use client'

import { useState, useCallback, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconVolume2 } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { useToastStore } from '@/lib/toast-store'

interface AudioPronunciationGuideProps {
  word?: string
  lang?: PhonemeLanguage
}

export default function AudioPronunciationGuide({ word = 'hello', lang = 'en' }: AudioPronunciationGuideProps) {
  const [loading, setLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleGenerate = useCallback(async () => {
    if (!word.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.synthesize(word, lang)
      const audioBlob = base64ToBlob(res.audio, 'audio/wav')
      const url = URL.createObjectURL(audioBlob)
      setAudioUrl(url)
      addToast('Audio generated!', 'success')
    } catch {
      addToast('Audio generation failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [word, lang, addToast])

  const handlePlay = useCallback(() => {
    if (!audioUrl) return
    if (audioRef.current) {
      audioRef.current.pause()
    }
    const audio = new Audio(audioUrl)
    audioRef.current = audio
    audio.onplay = () => setIsPlaying(true)
    audio.onended = () => setIsPlaying(false)
    audio.onerror = () => setIsPlaying(false)
    audio.play().catch(() => {
      addToast('Playback failed', 'error')
      setIsPlaying(false)
    })
  }, [audioUrl, addToast])

  const handleStop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
      setIsPlaying(false)
    }
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Audio Pronunciation</span>
          {audioUrl && (
            <Badge variant="default" className="bg-success">Ready</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Word:</span>
          <span className="font-medium">{word}</span>
          <Badge variant="outline">{PHONEME_LANGUAGES.find(l => l.value === lang)?.label}</Badge>
        </div>

        <div className="flex gap-3">
          <Button onClick={handleGenerate} disabled={loading || !word.trim()}>
            {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
            Generate Audio
          </Button>
          {audioUrl && (
            <>
              <Button onClick={isPlaying ? handleStop : handlePlay} variant="secondary">
                <IconVolume2 className="mr-2 h-4 w-4" />
                {isPlaying ? 'Stop' : 'Play'}
              </Button>
            </>
          )}
        </div>

        {audioUrl && (
          <div className="p-3 rounded-lg bg-muted/50">
            <audio ref={audioRef} src={audioUrl} className="w-full" controls />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function base64ToBlob(base64: string, type: string): Blob {
  const byteCharacters = atob(base64)
  const byteNumbers = new Array(byteCharacters.length)
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i)
  }
  const byteArray = new Uint8Array(byteNumbers)
  return new Blob([byteArray], { type })
}
