'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconCheck, IconX, IconBolt } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

interface ChallengeWord {
  word: string
  lang: PhonemeLanguage
  phonemes: string[]
  hint: string
  difficulty: 'easy' | 'medium' | 'hard'
  points: number
}

interface DailyChallenge {
  date: string
  words: ChallengeWord[]
  completed: boolean
  score: number
}

const CHALLENGE_WORDS: ChallengeWord[] = [
  // English challenges
  { word: 'think', lang: 'en', phonemes: ['TH', 'IH', 'NG', 'K'], hint: 'Dental fricative + velar nasal', difficulty: 'hard', points: 3 },
  { word: 'measure', lang: 'en', phonemes: ['M', 'EH', 'ZH', 'ER'], hint: 'Voiced postalveolar fricative', difficulty: 'hard', points: 3 },
  { word: 'three', lang: 'en', phonemes: ['TH', 'R', 'IY'], hint: 'Dental fricative + alveolar approximant', difficulty: 'medium', points: 2 },
  { word: 'church', lang: 'en', phonemes: ['CH', 'ER', 'CH'], hint: 'Postalveolar affricate', difficulty: 'medium', points: 2 },
  { word: 'hello', lang: 'en', phonemes: ['HH', 'AH', 'L', 'OW'], hint: 'Glottal fricative + diphthong', difficulty: 'easy', points: 1 },
  
  // German challenges
  { word: 'entschuldigen', lang: 'de', phonemes: ['EH', 'N', 'T', 'SH', 'U', 'L', 'D', 'I', 'G', 'AH', 'N'], hint: 'Complex consonant cluster', difficulty: 'hard', points: 3 },
  { word: 'ich', lang: 'de', phonemes: ['I', 'CH'], hint: 'Palatal fricative /ç/', difficulty: 'hard', points: 3 },
  { word: 'buch', lang: 'de', phonemes: ['B', 'U', 'CH'], hint: 'Velar fricative /x/ after back vowel', difficulty: 'medium', points: 2 },
  { word: 'Zug', lang: 'de', phonemes: ['TS', 'U', 'G'], hint: 'Affricate /ts/ + devoiced final', difficulty: 'medium', points: 2 },
  
  // French challenges
  { word: 'aujourd\'hui', lang: 'fr', phonemes: ['OH', 'ZH', 'U', 'R', 'D', 'U', 'I'], hint: 'Complex word with multiple syllables', difficulty: 'hard', points: 3 },
  { word: 'bonjour', lang: 'fr', phonemes: ['B', 'OH', 'N', 'ZH', 'U', 'R'], hint: 'Nasal vowel + voiced fricative', difficulty: 'medium', points: 2 },
  { word: 'croissant', lang: 'fr', phonemes: ['K', 'R', 'W', 'AA', 'S', 'AA', 'N'], hint: 'Diphthong + nasal', difficulty: 'medium', points: 2 },
  
  // Spanish challenges
  { word: 'ciudad', lang: 'es', phonemes: ['TH', 'I', 'U', 'D', 'AA', 'D'], hint: 'Dental fricative (Castilian)', difficulty: 'hard', points: 3 },
  { word: 'jardín', lang: 'es', phonemes: ['H', 'AA', 'R', 'D', 'I', 'N'], hint: 'Velar fricative /x/', difficulty: 'hard', points: 3 },
  { word: 'perro', lang: 'es', phonemes: ['P', 'EH', 'R', 'R', 'OH'], hint: 'Trilled /rr/', difficulty: 'medium', points: 2 },
  
  // Italian challenges
  { word: 'gnocchi', lang: 'it', phonemes: ['NY', 'OH', 'K', 'I'], hint: 'Palatal nasal /ɲ/', difficulty: 'hard', points: 3 },
  { word: 'ospedale', lang: 'it', phonemes: ['OH', 'S', 'P', 'EH', 'D', 'AA', 'L', 'EH'], hint: 'Voiced /z/ between vowels', difficulty: 'hard', points: 3 },
  { word: 'grazie', lang: 'it', phonemes: ['G', 'R', 'AA', 'TS', 'I', 'EH'], hint: 'Affricate /ts/', difficulty: 'medium', points: 2 },
  
  // Portuguese challenges
  { word: 'saudade', lang: 'pt', phonemes: ['S', 'AW', 'D', 'AA', 'D', 'UH'], hint: 'Nasal diphthong', difficulty: 'hard', points: 3 },
  { word: 'manhã', lang: 'pt', phonemes: ['M', 'AA', 'NY', 'AA', 'N'], hint: 'Palatal nasal + nasal vowel', difficulty: 'hard', points: 3 },
  { word: 'não', lang: 'pt', phonemes: ['N', 'AW', 'N'], hint: 'Nasal diphthong /ɐ̃w̃/', difficulty: 'hard', points: 3 },
]

const CHALLENGE_KEY = 'sloughgpt-phoneme-challenge'

function getToday(): string {
  return new Date().toISOString().split('T')[0]
}

function loadChallenge(): DailyChallenge | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(CHALLENGE_KEY)
    if (!raw) return null
    const data = JSON.parse(raw)
    if (data.date === getToday()) return data
    return null
  } catch {
    return null
  }
}

function saveChallenge(challenge: DailyChallenge) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(CHALLENGE_KEY, JSON.stringify(challenge))
  } catch {}
}

function generateDailyChallenge(): DailyChallenge {
  const today = getToday()
  const seed = today.split('-').join('')
  const rng = (s: number) => {
    let x = Math.sin(s) * 10000
    return x - Math.floor(x)
  }
  
  const shuffled = [...CHALLENGE_WORDS].sort((a, b) => {
    const hashA = seed.charCodeAt(0) * 31 + a.word.charCodeAt(0)
    const hashB = seed.charCodeAt(0) * 31 + b.word.charCodeAt(0)
    return rng(hashA) - rng(hashB)
  })
  
  const words = shuffled.slice(0, 5)
  
  return {
    date: today,
    words,
    completed: false,
    score: 0,
  }
}

export default function ChallengeCard() {
  const [challenge, setChallenge] = useState<DailyChallenge>(() => {
    return loadChallenge() || generateDailyChallenge()
  })
  const [currentWordIndex, setCurrentWordIndex] = useState(0)
  const [attempt, setAttempt] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ score: number; phonemes: string[] } | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const currentWord = challenge.words[currentWordIndex]

  const handleScore = useCallback(async () => {
    if (!currentWord || !attempt.trim()) return
    setLoading(true)
    try {
      const res = await phonemeController.score(currentWord.word, attempt, currentWord.lang)
      setResult({ score: res.score, phonemes: res.spoken_phonemes })
      
      const passed = res.score >= 0.7
      if (passed) {
        addToast(`+${currentWord.points} points!`, 'success')
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
  }, [currentWord, attempt, addToHistory, addToast])

  const handleNext = useCallback(() => {
    const passed = result && result.score >= 0.7
    const newScore = challenge.score + (passed ? currentWord.points : 0)
    
    if (currentWordIndex < challenge.words.length - 1) {
      setCurrentWordIndex(prev => prev + 1)
      setAttempt('')
      setResult(null)
    } else {
      const completed = true
      const finalChallenge = { ...challenge, score: newScore, completed }
      setChallenge(finalChallenge)
      saveChallenge(finalChallenge)
      
      if (newScore >= 10) {
        addToast('Challenge complete! Excellent!', 'success')
      } else if (newScore >= 5) {
        addToast('Challenge complete! Good job!', 'success')
      } else {
        addToast('Challenge complete! Keep practicing!', 'success')
      }
    }
  }, [currentWordIndex, challenge, currentWord, result, addToast])

  const handleNewChallenge = useCallback(() => {
    const newChallenge = generateDailyChallenge()
    setChallenge(newChallenge)
    saveChallenge(newChallenge)
    setCurrentWordIndex(0)
    setAttempt('')
    setResult(null)
  }, [])

  const totalPoints = challenge.words.reduce((sum, w) => sum + w.points, 0)
  const progress = ((currentWordIndex + 1) / challenge.words.length) * 100

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Daily Challenge</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{challenge.date}</Badge>
            {challenge.completed && (
              <Badge variant="default" className="bg-success">
                {challenge.score}/{totalPoints}
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {challenge.completed ? (
          <div className="text-center py-6 space-y-4">
            <div className="text-4xl font-bold">
              {challenge.score}/{totalPoints}
            </div>
            <p className="text-muted-foreground">
              {challenge.score >= totalPoints ? 'Perfect score!' :
               challenge.score >= totalPoints * 0.7 ? 'Great job!' :
               challenge.score >= totalPoints * 0.5 ? 'Good effort!' : 'Keep practicing!'}
            </p>
            <Button onClick={handleNewChallenge}>New Challenge</Button>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                Word {currentWordIndex + 1} of {challenge.words.length}
              </span>
              <span className="text-muted-foreground">
                {challenge.score} points
              </span>
            </div>
            
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>

            {currentWord && (
              <div className="space-y-3">
                <div className="p-4 rounded-lg bg-muted/50 text-center">
                  <p className="text-2xl font-bold">{currentWord.word}</p>
                  <div className="flex items-center justify-center gap-2 mt-2">
                    <Badge variant="outline">{PHONEME_LANGUAGES.find(l => l.value === currentWord.lang)?.label}</Badge>
                    <Badge variant={currentWord.difficulty === 'hard' ? 'destructive' : currentWord.difficulty === 'medium' ? 'secondary' : 'default'}>
                      {currentWord.difficulty}
                    </Badge>
                    <Badge variant="outline" className="gap-1">
                      <IconBolt className="h-3 w-3" />
                      {currentWord.points}
                    </Badge>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground text-center">
                  Hint: {currentWord.hint}
                </p>

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
                      Score
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className={`p-3 rounded-lg text-center ${result.score >= 0.7 ? 'bg-success/10' : 'bg-destructive/10'}`}>
                      <p className={`text-2xl font-bold ${result.score >= 0.7 ? 'text-success' : 'text-destructive'}`}>
                        {(result.score * 100).toFixed(0)}%
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {result.score >= 0.9 ? 'Excellent!' :
                         result.score >= 0.7 ? 'Great!' :
                         result.score >= 0.5 ? 'Good try!' : 'Keep practicing!'}
                      </p>
                    </div>
                    
                    <div className="flex flex-wrap gap-1 justify-center">
                      {result.phonemes.map((p, i) => (
                        <Badge
                          key={i}
                          variant={currentWord.phonemes[i] === p ? 'default' : 'destructive'}
                        >
                          {p}
                        </Badge>
                      ))}
                    </div>

                    <Button onClick={handleNext} className="w-full">
                      {currentWordIndex < challenge.words.length - 1 ? 'Next Word' : 'Complete Challenge'}
                    </Button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
