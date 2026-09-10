'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconMic, IconMicFilled, IconStop } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { Progress } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeScoreResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

const PRACTICE_WORDS: Record<PhonemeLanguage, string[]> = {
  en: ['hello', 'world', 'goodbye', 'please', 'thank', 'cat', 'dog', 'house', 'water', 'food'],
  de: ['hallo', 'welt', 'bitte', 'danke', 'ja', 'nein', 'katze', 'hund', 'haus', 'wasser'],
  fr: ['bonjour', 'monde', 'merci', 'oui', 'non', 'chat', 'chien', 'maison', 'eau', 'livre'],
  es: ['hola', 'mundo', 'gracias', 'si', 'no', 'gato', 'perro', 'casa', 'agua', 'libro'],
  it: ['ciao', 'mondo', 'grazie', 'si', 'no', 'gatto', 'casa', 'acqua', 'libro', 'macchina'],
  pt: ['ola', 'mundo', 'obrigado', 'sim', 'nao', 'gato', 'casa', 'agua', 'livro', 'carro'],
}

const PHONEME_TIPS: Record<string, string> = {
  TH: 'Place tongue between teeth, blow air',
  DH: 'Voiced version of TH — vibrate vocal cords',
  SH: 'Round lips, push tongue forward',
  ZH: 'Like SH but with voice — vibrate vocal cords',
  CH: 'Tongue touches roof of mouth, push air out',
  NG: 'Back of tongue touches soft palate',
  R: 'Curl tongue back without touching roof',
  L: 'Touch tongue tip to roof behind teeth',
  W: 'Round lips tightly',
  V: 'Bottom lip touches upper teeth',
  F: 'Bottom lip touches upper teeth',
  S: 'Tongue near roof, blow air through gap',
  Z: 'Like S but with voice',
  P: 'Press lips together, release burst of air',
  B: 'Like P but with voice',
  T: 'Tongue touches roof, release burst',
  D: 'Like T but with voice',
  K: 'Back of tongue touches roof, release',
  G: 'Like K but with voice',
  M: 'Press lips together, hum through nose',
  N: 'Tongue touches roof, air through nose',
  AE: 'Open mouth wide, short sound like "cat"',
  AH: 'Open mouth, neutral like "father"',
  EH: 'Open mouth slightly, like "bed"',
  EE: 'Stretch lips wide, long sound like "see"',
  OW: 'Round lips, long sound like "go"',
  UH: 'Relaxed mouth, short sound like "put"',
  UW: 'Round lips tightly, long sound like "food"',
  AY: 'Start open, slide to EE sound',
  AW: 'Start open, slide to OW sound',
  OY: 'Start with OW, slide to EE',
  IH: 'Short sound like "sit"',
  AO: 'Open mouth rounded like "caught"',
}

function getPhonemeTip(phoneme: string): string | null {
  return PHONEME_TIPS[phoneme] ?? null
}

export default function PracticeCard() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [targetWord, setTargetWord] = useState('')
  const [attempt, setAttempt] = useState('')
  const [targetResult, setTargetResult] = useState<PhonemeEncodeResult | null>(null)
  const [attemptResult, setAttemptResult] = useState<PhonemeEncodeResult | null>(null)
  const [scoreResult, setScoreResult] = useState<PhonemeScoreResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const attemptDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const randomWordTrigger = usePhonemeStore(s => s.randomWordTrigger)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (attemptDebounceRef.current) clearTimeout(attemptDebounceRef.current)
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  const fetchTarget = useCallback(async (word: string) => {
    if (!word.trim()) return
    try {
      const res = await phonemeController.encode(word, language)
      setTargetResult(res)
    } catch {
      setTargetResult(null)
    }
  }, [language])

  const fetchAttempt = useCallback(async (word: string) => {
    if (!word.trim()) { setAttemptResult(null); return }
    try {
      const res = await phonemeController.encode(word, language)
      setAttemptResult(res)
    } catch {
      setAttemptResult(null)
    }
  }, [language])

  const handleWordChange = useCallback((word: string) => {
    setTargetWord(word)
    setScoreResult(null)
    setAttempt('')
    setAttemptResult(null)
    fetchTarget(word)
  }, [fetchTarget])

  useEffect(() => {
    if (randomWordTrigger > 0) {
      const words = PRACTICE_WORDS[language]
      const word = words[Math.floor(Math.random() * words.length)]
      handleWordChange(word)
    }
  }, [randomWordTrigger, language, handleWordChange])

  const handleAttemptChange = useCallback((word: string) => {
    setAttempt(word)
    setScoreResult(null)
    if (attemptDebounceRef.current) clearTimeout(attemptDebounceRef.current)
    attemptDebounceRef.current = setTimeout(() => fetchAttempt(word), 300)
  }, [fetchAttempt])

  const handleRandomWord = useCallback(() => {
    const words = PRACTICE_WORDS[language]
    const word = words[Math.floor(Math.random() * words.length)]
    handleWordChange(word)
  }, [language, handleWordChange])

  const handleScore = useCallback(async () => {
    if (!targetWord.trim() || !attempt.trim()) return
    if (attemptDebounceRef.current) clearTimeout(attemptDebounceRef.current)
    setLoading(true)
    try {
      const res = await phonemeController.score(targetWord, attempt, language)
      setScoreResult(res)
      addToHistory({
        target: res.target,
        spoken: res.spoken,
        score: res.score,
        language: res.language,
        targetPhonemes: res.target_phonemes,
        spokenPhonemes: res.spoken_phonemes,
      })
    } catch {
      addToast('Scoring failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [targetWord, attempt, language, addToHistory, addToast])

  const toggleRecording = useCallback(async () => {
    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []

      recorder.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `recording-${Date.now()}.wav`
        a.click()
        URL.revokeObjectURL(url)
      }

      recorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1)
      }, 1000)
    } catch {
      addToast('Microphone access denied', 'error')
    }
  }, [isRecording, addToast])

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Practice Mode</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PHONEME_LANGUAGES.map(l => (
                  <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="secondary" onClick={handleRandomWord}>Random Word</Button>
          </div>

          <div className="flex gap-3">
            <Input
              value={targetWord}
              onChange={e => handleWordChange(e.target.value)}
              placeholder="Enter word to practice..."
              onKeyDown={e => e.key === 'Enter' && fetchTarget(targetWord)}
              className="flex-1"
            />
          </div>

          {targetResult && (
            <div className="p-3 rounded-lg bg-muted/50 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Target:</span>
                <span className="font-bold">{targetResult.text}</span>
                <Badge variant="secondary">{targetResult.language.toUpperCase()}</Badge>
              </div>
              <div className="flex flex-wrap gap-1">
                {targetResult.phonemes.map((p, i) => (
                  <Badge key={i} variant="outline">{p}</Badge>
                ))}
              </div>

              <div className="mt-3 space-y-1">
                <p className="text-sm font-medium text-muted-foreground">Pronunciation Tips:</p>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {targetResult.phonemes.map((p, i) => {
                    const tip = getPhonemeTip(p)
                    return tip ? (
                      <li key={i}><Badge variant="outline" className="mr-1">{p}</Badge>{tip}</li>
                    ) : null
                  })}
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Attempt</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              value={attempt}
              onChange={e => handleAttemptChange(e.target.value)}
              placeholder="Type how you would say it..."
              onKeyDown={e => e.key === 'Enter' && handleScore()}
              className="flex-1"
            />
            <Button onClick={handleScore} disabled={loading || !targetWord.trim() || !attempt.trim()}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Score
            </Button>
            <Button
              variant={isRecording ? 'destructive' : 'secondary'}
              onClick={toggleRecording}
            >
              {isRecording ? <IconStop className="mr-2 h-4 w-4" /> : <IconMic className="mr-2 h-4 w-4" />}
              {isRecording ? `Stop (${recordingTime}s)` : 'Record'}
            </Button>
          </div>

          {attemptResult && (
            <div className="p-3 rounded-lg bg-muted/50 space-y-1 animate-in fade-in slide-in-from-top-1 duration-200">
              <p className="text-sm font-medium text-muted-foreground">Your phonemes:</p>
              <div className="flex flex-wrap gap-1">
                {attemptResult.phonemes.map((p, i) => (
                  <Badge key={i} variant="outline">{p}</Badge>
                ))}
              </div>
            </div>
          )}

          {scoreResult && (
            <div className="p-4 rounded-lg bg-muted/50 space-y-3 animate-in fade-in slide-in-from-top-1 duration-200">
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold">{(scoreResult.score * 100).toFixed(0)}%</span>
                <div className="flex gap-4 text-sm text-muted-foreground">
                  <span>Precision: {(scoreResult.precision * 100).toFixed(0)}%</span>
                  <span>Recall: {(scoreResult.recall * 100).toFixed(0)}%</span>
                </div>
              </div>

              <Progress value={scoreResult.score * 100} />

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Target:</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {scoreResult.target_phonemes.map((p, i) => (
                      <Badge key={i} variant="outline">{p}</Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Your attempt:</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {scoreResult.spoken_phonemes.map((p, i) => (
                      <Badge
                        key={i}
                        variant={scoreResult.target_phonemes[i] === p ? 'default' : 'destructive'}
                      >
                        {p}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
