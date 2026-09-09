'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

const WOTD_WORDS: { word: string; lang: PhonemeLanguage }[] = [
  { word: 'ephemeral', lang: 'en' },
  { word: 'sonder', lang: 'en' },
  { word: 'petrichor', lang: 'en' },
  { word: 'serendipity', lang: 'en' },
  { word: 'Wanderlust', lang: 'de' },
  { word: 'Schadenfreude', lang: 'de' },
  { word: 'Zeitgeist', lang: 'de' },
  { word: 'joie de vivre', lang: 'fr' },
  { word: 'rendez-vous', lang: 'fr' },
  { word: 'sobremesa', lang: 'es' },
  { word: 'duende', lang: 'es' },
  { word: 'saudade', lang: 'pt' },
  { word: 'campanile', lang: 'it' },
]

export default function QuizCard() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [guess, setGuess] = useState('')
  const [currentWord, setCurrentWord] = useState<typeof WOTD_WORDS[number] | null>(null)
  const [currentPhonemes, setCurrentPhonemes] = useState<string[]>([])
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | 'revealed' | null>(null)
  const [revealedWord, setRevealedWord] = useState('')
  const [loading, setLoading] = useState(false)
  const quizScore = usePhonemeStore(s => s.quizScore)
  const quizTotal = usePhonemeStore(s => s.quizTotal)
  const incrementQuizScore = usePhonemeStore(s => s.incrementQuizScore)
  const addToast = useToastStore(s => s.addToast)

  const startQuiz = useCallback(async () => {
    const filtered = WOTD_WORDS.filter(w => w.lang === language)
    const pick = filtered.length > 0
      ? filtered[Math.floor(Math.random() * filtered.length)]
      : WOTD_WORDS[Math.floor(Math.random() * WOTD_WORDS.length)]

    setCurrentWord(pick)
    setFeedback(null)
    setGuess('')
    setRevealedWord('')

    try {
      const res = await phonemeController.encode(pick.word, pick.lang)
      setCurrentPhonemes(res.phonemes)
    } catch {
      setCurrentPhonemes([])
    }
  }, [language])

  const handleGuess = useCallback(() => {
    if (!currentWord || !guess.trim()) return
    const correct = guess.trim().toLowerCase() === currentWord.word.toLowerCase()
    incrementQuizScore(correct)
    setFeedback(correct ? 'correct' : 'incorrect')
    setRevealedWord(currentWord.word)
  }, [currentWord, guess, incrementQuizScore])

  const handleReveal = useCallback(() => {
    if (!currentWord) return
    incrementQuizScore(false)
    setFeedback('revealed')
    setRevealedWord(currentWord.word)
  }, [currentWord, incrementQuizScore])

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Word of the Day</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {currentWord ? (
            <>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold">{currentWord.word}</span>
                <Badge variant="secondary">{currentWord.lang.toUpperCase()}</Badge>
              </div>
              {currentPhonemes.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {currentPhonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
              )}
              <div className="flex gap-2">
                <Button variant="secondary" onClick={startQuiz}>New Word</Button>
                <Button onClick={() => {
                  setRevealedWord(currentWord.word)
                  setFeedback(null)
                }}>Reveal</Button>
              </div>
            </>
          ) : (
            <p className="text-muted-foreground">Click Start Quiz to begin</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Quiz</span>
            <Badge variant="secondary">{quizScore}/{quizTotal}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3">
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
            <Button onClick={startQuiz} disabled={loading}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Start Quiz
            </Button>
          </div>

          {currentWord && (
            <>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-sm text-muted-foreground">Hint — Phonemes:</p>
                <div className="flex flex-wrap gap-1 justify-center mt-1">
                  {currentPhonemes.map((p, i) => (
                    <Badge key={i} variant="outline">?</Badge>
                  ))}
                </div>
              </div>

              <div className="flex gap-3">
                <Input
                  value={guess}
                  onChange={e => setGuess(e.target.value)}
                  placeholder="Type the word..."
                  onKeyDown={e => e.key === 'Enter' && handleGuess()}
                  className="flex-1"
                />
                <Button onClick={handleGuess} disabled={!guess.trim()}>Check</Button>
                <Button variant="secondary" onClick={handleReveal}>Reveal</Button>
              </div>

              {feedback && (
                <div className={`p-3 rounded-lg text-center ${
                  feedback === 'correct' ? 'bg-green-500/10 text-green-500' :
                  feedback === 'incorrect' ? 'bg-red-500/10 text-red-500' :
                  'bg-yellow-500/10 text-yellow-500'
                }`}>
                  {feedback === 'correct' && <p><strong>Correct!</strong> The answer was &quot;{revealedWord}&quot;</p>}
                  {feedback === 'incorrect' && <p><strong>Incorrect.</strong> The answer was &quot;{revealedWord}&quot;</p>}
                  {feedback === 'revealed' && <p><strong>Revealed.</strong> The answer was &quot;{revealedWord}&quot;</p>}
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
