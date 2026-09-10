'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconBolt } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeEncodeResult, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

type Difficulty = 'easy' | 'medium' | 'hard'

const WOTD_WORDS: { word: string; lang: PhonemeLanguage; difficulty: Difficulty }[] = [
  { word: 'ephemeral', lang: 'en', difficulty: 'hard' },
  { word: 'sonder', lang: 'en', difficulty: 'medium' },
  { word: 'petrichor', lang: 'en', difficulty: 'hard' },
  { word: 'serendipity', lang: 'en', difficulty: 'medium' },
  { word: 'Wanderlust', lang: 'de', difficulty: 'easy' },
  { word: 'Schadenfreude', lang: 'de', difficulty: 'hard' },
  { word: 'Zeitgeist', lang: 'de', difficulty: 'medium' },
  { word: 'joie de vivre', lang: 'fr', difficulty: 'hard' },
  { word: 'rendez-vous', lang: 'fr', difficulty: 'medium' },
  { word: 'sobremesa', lang: 'es', difficulty: 'medium' },
  { word: 'duende', lang: 'es', difficulty: 'hard' },
  { word: 'saudade', lang: 'pt', difficulty: 'hard' },
  { word: 'campanile', lang: 'it', difficulty: 'medium' },
]

export default function QuizCard() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [difficulty, setDifficulty] = useState<Difficulty>('medium')
  const [guess, setGuess] = useState('')
  const [currentWord, setCurrentWord] = useState<typeof WOTD_WORDS[number] | null>(null)
  const [currentPhonemes, setCurrentPhonemes] = useState<string[]>([])
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | 'revealed' | null>(null)
  const [revealedWord, setRevealedWord] = useState('')
  const [loading, setLoading] = useState(false)
  const quizScore = usePhonemeStore(s => s.quizScore)
  const quizTotal = usePhonemeStore(s => s.quizTotal)
  const quizStreak = usePhonemeStore(s => s.quizStreak)
  const quizBestStreak = usePhonemeStore(s => s.quizBestStreak)
  const incrementQuizScore = usePhonemeStore(s => s.incrementQuizScore)
  const resetQuiz = usePhonemeStore(s => s.resetQuiz)
  const addToast = useToastStore(s => s.addToast)

  const startQuiz = useCallback(async () => {
    const filtered = WOTD_WORDS.filter(w => w.lang === language && w.difficulty === difficulty)
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
  }, [language, difficulty])

  const handleGuess = useCallback(() => {
    if (!currentWord || !guess.trim()) return
    const correct = guess.trim().toLowerCase() === currentWord.word.toLowerCase()
    incrementQuizScore(correct)
    setFeedback(correct ? 'correct' : 'incorrect')
    setRevealedWord(currentWord.word)
    if (correct && quizStreak >= 2) {
      addToast(`${quizStreak + 1} streak!`, 'success')
    }
  }, [currentWord, guess, incrementQuizScore, quizStreak, addToast])

  const handleReveal = useCallback(() => {
    if (!currentWord) return
    incrementQuizScore(false)
    setFeedback('revealed')
    setRevealedWord(currentWord.word)
  }, [currentWord, incrementQuizScore])

  const showPhonemeHint = difficulty === 'easy'
  const showPhonemeCount = difficulty !== 'hard'

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Word of the Day</span>
            {quizStreak >= 3 && (
              <Badge variant="default" className="gap-1 bg-warning text-warning-foreground">
                <IconBolt className="h-3 w-3" />
                {quizStreak}
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {currentWord ? (
            <>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold">{currentWord.word}</span>
                <Badge variant="secondary">{currentWord.lang.toUpperCase()}</Badge>
                <Badge variant="outline" className="text-xs">{currentWord.difficulty}</Badge>
              </div>
              {showPhonemeHint && currentPhonemes.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {currentPhonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
              )}
              {!showPhonemeHint && showPhonemeCount && currentPhonemes.length > 0 && (
                <p className="text-sm text-muted-foreground">{currentPhonemes.length} phonemes</p>
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
            <div className="flex items-center gap-2">
              {quizStreak >= 3 && (
                <span className="text-sm text-warning font-medium flex items-center gap-1">
                  <IconBolt className="h-3 w-3" />{quizStreak}
                </span>
              )}
              <Badge variant="secondary">{quizScore}/{quizTotal}</Badge>
            </div>
          </CardTitle>
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
            <Select value={difficulty} onValueChange={v => setDifficulty(v as Difficulty)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="easy">Easy</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="hard">Hard</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={startQuiz} disabled={loading}>
              {loading ? <IconRefresh className="animate-spin mr-2 h-4 w-4" /> : null}
              Start Quiz
            </Button>
            {quizTotal > 0 && (
              <Button variant="ghost" size="sm" onClick={resetQuiz}>Reset</Button>
            )}
          </div>

          {currentWord && (
            <>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-sm text-muted-foreground">Hint — Phonemes:</p>
                <div className="flex flex-wrap gap-1 justify-center mt-1">
                  {currentPhonemes.map((_, i) => (
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
                <div className={`p-3 rounded-lg text-center animate-in fade-in duration-200 ${
                  feedback === 'correct' ? 'bg-success/10 text-success' :
                  feedback === 'incorrect' ? 'bg-destructive/10 text-destructive' :
                  'bg-warning/10 text-warning'
                }`}>
                  {feedback === 'correct' && <p><strong>Correct!</strong> The answer was &quot;{revealedWord}&quot;</p>}
                  {feedback === 'incorrect' && <p><strong>Incorrect.</strong> The answer was &quot;{revealedWord}&quot;</p>}
                  {feedback === 'revealed' && <p><strong>Revealed.</strong> The answer was &quot;{revealedWord}&quot;</p>}
                </div>
              )}
            </>
          )}

          {quizBestStreak > 0 && (
            <p className="text-xs text-muted-foreground text-center">Best streak: {quizBestStreak}</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
