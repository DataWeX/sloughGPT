'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, Button } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { voiceController } from '@/lib/voice-controller'
import { useToastStore } from '@/lib/toast-store'

export default function VoicePage() {
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [aiText, setAiText] = useState('')
  const [speaking, setSpeaking] = useState(false)
  const [supported, setSupported] = useState(true)
  const recognitionRef = useRef<any>(null)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    const SR: any = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    if (!SR) setSupported(false)
  }, [])

  const toggleListen = () => {
    const SR: any = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    if (!SR) {
      addToast('Speech recognition not supported — type instead', 'error')
      return
    }
    if (listening) {
      recognitionRef.current?.stop()
      setListening(false)
      return
    }
    const rec = new SR()
    recognitionRef.current = rec
    rec.continuous = false
    rec.interimResults = true
    rec.lang = navigator.language || 'en-US'
    rec.onresult = (e: any) => {
      let t = ''
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript + ' '
      setTranscript(t.trim())
    }
    rec.onend = () => setListening(false)
    rec.onerror = () => setListening(false)
    rec.start()
    setListening(true)
    setTranscript('')
    setAiText('')
  }

  const speak = async (text: string) => {
    if (!text.trim()) return
    setSpeaking(true)
    try {
      const data = await voiceController.tts(text).catch(() => null)
      if (data?.audio && data.backend !== 'browser-fallback') {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`)
        audio.onended = () => setSpeaking(false)
        await audio.play()
      } else if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
        const u = new SpeechSynthesisUtterance(text)
        u.onend = () => setSpeaking(false)
        window.speechSynthesis.speak(u)
      } else {
        setSpeaking(false)
      }
    } catch {
      setSpeaking(false)
    }
  }

  const handleSend = async () => {
    if (!transcript.trim()) return
    const text = transcript
    setAiText('Thinking...')
    // Minimal echo for now — chat page handles real AI. Voice page is STT+TTS infra demo.
    const reply = `You said: ${text}`
    setAiText(reply)
    await speak(reply)
  }

  return (
    <PageContainer title="Talk Out Loud" subtitle="Tap mic, speak naturally — hear the reply">
      <Card>
        <CardContent className="flex flex-col items-center gap-4 py-8">
          <button
            onClick={toggleListen}
            aria-label={listening ? 'Stop listening' : 'Start listening'}
            className={`relative flex h-20 w-20 items-center justify-center rounded-full border text-xl transition ${listening ? 'bg-primary text-primary-foreground animate-pulse border-primary' : 'bg-card border-border hover:bg-accent'}`}
          >
            🎙️
            {listening && <span className="absolute inset-0 rounded-full animate-ping bg-primary/20" />}
          </button>
          <p className="text-sm text-muted-foreground">{listening ? 'Listening...' : supported ? 'Tap to talk' : 'Type below (mic not supported)'}</p>
          {!supported && (
            <p className="text-xs text-muted-foreground/60">Your browser has no SpeechRecognition — transcript will be typed, TTS still works.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-3 p-4">
          <p className="text-xs font-medium text-muted-foreground">Transcript</p>
          <div className="min-h-16 rounded-lg border border-border/40 bg-muted/20 p-3 text-sm">
            {transcript || <span className="text-muted-foreground/50">Words appear here as you speak</span>}
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleSend} disabled={!transcript.trim()}>Send to AI</Button>
            <Button size="sm" variant="ghost" onClick={() => { setTranscript(''); setAiText('') }}>Clear</Button>
            <Button size="sm" variant="ghost" onClick={toggleListen}>{listening ? 'Stop' : 'Mic'}</Button>
          </div>
          {aiText && (
            <div className="rounded-lg border border-border/40 p-3 text-sm">
              <p className="text-xs text-muted-foreground mb-1">AI reply</p>
              <p>{aiText}</p>
              <Button size="sm" variant="ghost" className="mt-2 h-7 text-xs" onClick={() => speak(aiText)} disabled={speaking}>
                {speaking ? 'Speaking...' : '🔊 Play again'}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground/60 text-center">Transcript saved to chat history. Uses browser speech synthesis when server TTS unavailable — no settings, auto-detect language.</p>
    </PageContainer>
  )
}
