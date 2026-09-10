'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconCheck, IconX, IconBolt } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

interface WordOfDay {
  word: string
  lang: PhonemeLanguage
  phonemes: string[]
  ipa: string
  definition: string
  tip: string
  difficulty: 'easy' | 'medium' | 'hard'
}

const WORDS_OF_DAY: WordOfDay[] = [
  // English
  { word: 'ephemeral', lang: 'en', phonemes: ['IH', 'F', 'EH', 'M', 'ER', 'AH', 'L'], ipa: '/ɪˈfɛmərəl/', definition: 'lasting for a very short time', tip: 'The /f/ is labiodental. Stress on second syllable. The /ə/ schwa is unstressed.', difficulty: 'hard' },
  { word: 'serendipity', lang: 'en', phonemes: ['S', 'EH', 'R', 'AH', 'N', 'D', 'IH', 'P', 'IH', 'T', 'IY'], ipa: '/ˌsɛrənˈdɪpɪti/', definition: 'the occurrence of events by chance in a happy way', tip: 'Stress on third syllable. The /ð/ is not present - use /d/.', difficulty: 'hard' },
  { word: 'quintessential', lang: 'en', phonemes: ['K', 'W', 'IH', 'N', 'T', 'IH', 'S', 'EH', 'N', 'SH', 'AH', 'L'], ipa: '/ˌkwɪntɪˈsɛnʃəl/', definition: 'representing the most perfect or typical example', tip: 'The /kw/ cluster. Stress on third syllable. The /ʃ/ is postalveolar.', difficulty: 'hard' },
  { word: 'mellifluous', lang: 'en', phonemes: ['M', 'AH', 'L', 'IH', 'F', 'L', 'UW', 'AH', 'S'], ipa: '/məˈlɪfluəs/', definition: 'sweet-sounding; pleasant to hear', tip: 'The /fl/ cluster. Stress on second syllable. The /u/ is a glide.', difficulty: 'medium' },
  { word: 'petrichor', lang: 'en', phonemes: ['P', 'EH', 'T', 'R', 'IH', 'K', 'ER'], ipa: '/ˈpɛtrɪkɔːr/', definition: 'the pleasant smell after rain', tip: 'The /tr/ cluster. Stress on first syllable. The /ɔ/ is open-mid back rounded.', difficulty: 'medium' },
  
  // German
  { word: 'Wanderlust', lang: 'de', phonemes: ['V', 'AA', 'N', 'D', 'ER', 'L', 'U', 'S', 'T'], ipa: '/ˈvandɐlʊst/', definition: 'a strong desire to travel', tip: 'German /v/ is labiodental fricative. The /ɐ/ is a reduced vowel.', difficulty: 'medium' },
  { word: 'Zeitgeist', lang: 'de', phonemes: ['TS', 'AY', 'T', 'G', 'AY', 'S', 'T'], ipa: '/ˈtsaɪtɡaɪst/', definition: 'the defining spirit or mood of a particular period', tip: 'The /ts/ affricate. The diphthong /aɪ/ glides from open to close front.', difficulty: 'medium' },
  { word: 'Schadenfreude', lang: 'de', phonemes: ['SH', 'AA', 'D', 'AH', 'N', 'F', 'R', 'OY', 'D', 'AH'], ipa: '/ˈʃɑːdənfrɔɪdə/', definition: 'pleasure derived from another\'s misfortune', tip: 'The /ʃ/ postalveolar. The diphthong /ɔʏ/ rounds then spreads.', difficulty: 'hard' },
  
  // French
  { word: 'l\'esprit', lang: 'fr', phonemes: ['L', 'EH', 'S', 'P', 'R', 'I'], ipa: '/lɛspʁi/', definition: 'the mind or spirit', tip: 'The /ʁ/ uvular fricative. The /i/ is close front unrounded.', difficulty: 'medium' },
  { word: 'dépaysement', lang: 'fr', phonemes: ['D', 'EH', 'P', 'AY', 'Z', 'AH', 'M', 'AA', 'N'], ipa: '/depɛizmɑ̃/', definition: 'the feeling of being a foreigner', tip: 'The nasal /ɑ̃/ is produced with velum lowered. The /z/ is voiced.', difficulty: 'hard' },
  { word: 'joie', lang: 'fr', phonemes: ['ZH', 'W', 'AA'], ipa: '/ʒwa/', definition: 'joy', tip: 'The /ʒ/ is voiced postalveolar fricative. The /wa/ diphthong glides.', difficulty: 'easy' },
  
  // Spanish
  { word: 'sobremesa', lang: 'es', phonemes: ['S', 'OH', 'B', 'R', 'EH', 'M', 'EH', 'S', 'AH'], ipa: '/soβɾemeˈsa/', definition: 'time spent talking after a meal', tip: 'The /β/ is a bilabial fricative (softer than /b/). Stress on penultimate.', difficulty: 'medium' },
  { word: 'duende', lang: 'es', phonemes: ['D', 'U', 'EH', 'N', 'D', 'EH'], ipa: '/dwenˈde/', definition: 'a quality of passion and inspiration', tip: 'The /dwen/ cluster. The /e/ is close-mid front unrounded.', difficulty: 'medium' },
  { word: 'trasnochar', lang: 'es', phonemes: ['T', 'R', 'AA', 'S', 'N', 'OH', 'CH', 'AA', 'R'], ipa: '/tɾasnoˈtʃar/', definition: 'to stay up late', tip: 'The /tr/ cluster. The /tʃ/ affricate is voiceless.', difficulty: 'hard' },
  
  // Italian
  { word: 'la dolce vita', lang: 'it', phonemes: ['L', 'AA', 'D', 'OH', 'L', 'CH', 'EH', 'V', 'I', 'T', 'AH'], ipa: '/la ˈdoltʃe ˈvita/', definition: 'the sweet life', tip: 'The /ltʃ/ cluster. The /v/ is labiodental fricative.', difficulty: 'medium' },
  { word: 'campanile', lang: 'it', phonemes: ['K', 'AA', 'M', 'P', 'AA', 'N', 'I', 'L', 'EH'], ipa: '/kampaniˈle/', definition: 'bell tower', tip: 'The /mp/ cluster. Stress on penultimate syllable.', difficulty: 'medium' },
  { word: 'farfalla', lang: 'it', phonemes: ['F', 'AA', 'R', 'F', 'AA', 'L', 'L', 'AH'], ipa: '/farˈfalla/', definition: 'butterfly', tip: 'Double /f/ and /l/ are geminated. Hold them longer.', difficulty: 'medium' },
  
  // Portuguese
  { word: 'saudade', lang: 'pt', phonemes: ['S', 'AW', 'D', 'AA', 'D', 'UH'], ipa: '/sawˈdadɨ/', definition: 'deep emotional state of nostalgic longing', tip: 'The nasal diphthong /ɐ̃w̃/ is unique to Portuguese. The final /d/ is dental.', difficulty: 'hard' },
  { word: 'cafuné', lang: 'pt', phonemes: ['K', 'AA', 'F', 'U', 'N', 'EH'], ipa: '/kafuˈne/', definition: 'a tender gesture of stroking someone\'s hair', tip: 'The /f/ is labiodental. The /u/ is close back rounded.', difficulty: 'medium' },
  { word: 'desenrascar', lang: 'pt', phonemes: ['D', 'EH', 'Z', 'EH', 'N', 'R', 'AA', 'S', 'K', 'AA', 'R'], ipa: '/dẽzẽɾaʃˈkaɾ/', definition: 'to figure something out', tip: 'The /z/ is voiced. The nasal /ẽ/ is mid front.', difficulty: 'hard' },
]

const WORD_KEY = 'sloughgpt-phoneme-word-of-day'

function getToday(): string {
  return new Date().toISOString().split('T')[0]
}

function loadWordOfDay(): { word: WordOfDay; practiced: boolean } | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(WORD_KEY)
    if (!raw) return null
    const data = JSON.parse(raw)
    if (data.date === getToday()) return data
    return null
  } catch {
    return null
  }
}

function saveWordOfDay(data: { date: string; word: WordOfDay; practiced: boolean }) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(WORD_KEY, JSON.stringify(data))
  } catch {}
}

function getWordOfDay(): WordOfDay {
  const today = getToday()
  const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000)
  return WORDS_OF_DAY[dayOfYear % WORDS_OF_DAY.length]
}

export default function WordOfDayCard() {
  const [state, setState] = useState(() => {
    const stored = loadWordOfDay()
    if (stored) return stored
    const word = getWordOfDay()
    return { word, practiced: false }
  })
  const [attempt, setAttempt] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ score: number; phonemes: string[] } | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const handleScore = useCallback(async () => {
    if (!state.word || !attempt.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.score(state.word.word, attempt, state.word.lang)
      setResult({ score: res.score, phonemes: res.spoken_phonemes })
      
      const passed = res.score >= 0.7
      if (passed) {
        setState(prev => ({ ...prev, practiced: true }))
        saveWordOfDay({ date: getToday(), word: state.word, practiced: true })
        addToast('Word of the day completed!', 'success')
      }
      
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
  }, [state.word, attempt, addToHistory, addToast])

  const handleNewWord = useCallback(() => {
    const word = getWordOfDay()
    setState({ word, practiced: false })
    setAttempt('')
    setResult(null)
    saveWordOfDay({ date: getToday(), word, practiced: false })
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Word of the Day</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{getToday()}</Badge>
            {state.practiced && (
              <Badge variant="default" className="bg-success">Practiced</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-4 rounded-lg bg-muted/50 text-center space-y-2">
          <p className="text-3xl font-bold">{state.word.word}</p>
          <p className="text-lg text-muted-foreground font-mono">{state.word.ipa}</p>
          <p className="text-sm">{state.word.definition}</p>
          <div className="flex items-center justify-center gap-2">
            <Badge variant="outline">{PHONEME_LANGUAGES.find(l => l.value === state.word.lang)?.label}</Badge>
            <Badge variant={state.word.difficulty === 'hard' ? 'destructive' : state.word.difficulty === 'medium' ? 'secondary' : 'default'}>
              {state.word.difficulty}
            </Badge>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-primary/5 border border-primary/10">
          <p className="text-sm"><span className="font-medium">Pronunciation tip:</span> {state.word.tip}</p>
        </div>

        <div className="flex flex-wrap gap-1 justify-center">
          {state.word.phonemes.map((p, i) => (
            <Badge key={i} variant="outline">{p}</Badge>
          ))}
        </div>

        {!result ? (
          <div className="flex gap-3">
            <input
              value={attempt}
              onChange={e => setAttempt(e.target.value)}
              placeholder="Type your pronunciation..."
              className="flex-1 px-3 py-2 rounded-md border bg-background text-sm"
              onKeyDown={e => e.key === 'Enter' && handleScore()}
            />
            <Button onClick={handleScore} disabled={loading || !attempt.trim()}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Practice
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className={`p-4 rounded-lg text-center ${result.score >= 0.7 ? 'bg-success/10' : 'bg-destructive/10'}`}>
              <p className={`text-3xl font-bold ${result.score >= 0.7 ? 'text-success' : 'text-destructive'}`}>
                {(result.score * 100).toFixed(0)}%
              </p>
              <p className="text-sm text-muted-foreground">
                {result.score >= 0.9 ? 'Excellent!' :
                 result.score >= 0.7 ? 'Great job!' :
                 result.score >= 0.5 ? 'Good try!' : 'Keep practicing!'}
              </p>
            </div>
            
            <div className="flex flex-wrap gap-1 justify-center">
              {result.phonemes.map((p, i) => (
                <Badge
                  key={i}
                  variant={state.word.phonemes[i] === p ? 'default' : 'destructive'}
                >
                  {p}
                </Badge>
              ))}
            </div>

            <Button onClick={handleNewWord} variant="secondary" className="w-full">
              New Word
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
