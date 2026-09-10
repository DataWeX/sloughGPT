'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import PhonemeSkeleton from './PhonemeSkeleton'

const PATTERNS = [
  { pattern: 'CVC', label: 'Consonant-Vowel-Consonant', examples: ['cat', 'dog', 'run'] },
  { pattern: 'CVCC', label: 'Consonant-Vowel-Consonant-Consonant', examples: ['best', 'hand', 'jump'] },
  { pattern: 'VCCV', label: 'Vowel-Consonant-Consonant-Vowel', examples: ['apple', 'happy', 'mirror'] },
  { pattern: 'CVCV', label: 'Consonant-Vowel-Consonant-Vowel', examples: ['baby', 'kite', 'nose'] },
  { pattern: 'CCVC', label: 'Consonant-Consonant-Vowel-Consonant', examples: ['stop', 'blue', 'tree'] },
]

interface QuizQuestion {
  word: string
  phonemes: string[]
  correctPattern: string
  options: string[]
}

type GameState = 'idle' | 'playing' | 'revealed'

function classifyPhoneme(p: string): string {
  const vowels = ['IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AH', 'AO', 'OW', 'OY', 'UH', 'UW', 'ER', 'AX']
  return vowels.includes(p) ? 'V' : 'C'
}

function classifyWord(phonemes: string[]): string {
  return phonemes.map(classifyPhoneme).join('')
}

export default function PhonemePatternQuiz() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [gameState, setGameState] = useState<GameState>('idle')
  const [question, setQuestion] = useState<QuizQuestion | null>(null)
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const [streak, setStreak] = useState(0)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const generateQuestion = useCallback(async () => {
    const patternPool = PATTERNS[Math.floor(Math.random() * PATTERNS.length)]
    const word = patternPool.examples[Math.floor(Math.random() * patternPool.examples.length)]

    try {
      const result = await phonemeController.encode(word, language)
      const correctPattern = classifyWord(result.phonemes)

      const otherPatterns = PATTERNS
        .filter(p => p.pattern !== correctPattern)
        .sort(() => Math.random() - 0.5)
        .slice(0, 3)
        .map(p => p.pattern)

      const options = [correctPattern, ...otherPatterns].sort(() => Math.random() - 0.5)

      setQuestion({
        word,
        phonemes: result.phonemes,
        correctPattern,
        options,
      })
      setGameState('playing')
    } catch {
      addToast('Failed to encode word', 'error')
    }
  }, [language, addToast])

  const startGame = useCallback(() => {
    setScore(0)
    setTotal(0)
    setStreak(0)
    generateQuestion()
  }, [generateQuestion])

  const checkAnswer = useCallback((selected: string) => {
    if (!question) return

    const isCorrect = selected === question.correctPattern
    setTotal(t => t + 1)
    if (isCorrect) {
      setScore(s => s + 1)
      setStreak(s => s + 1)
    } else {
      setStreak(0)
    }
    setGameState('revealed')

    addToHistory({
      target: question.word,
      spoken: question.word,
      targetPhonemes: question.phonemes,
      spokenPhonemes: question.phonemes,
      score: isCorrect ? 1.0 : 0.3,
      language,
    })

    addToast(
      isCorrect ? `Correct! ${question.word} is ${question.correctPattern}` : `The pattern was ${question.correctPattern}`,
      isCorrect ? 'success' : 'error',
    )
  }, [question, language, addToHistory, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pattern Recognition Quiz</span>
          {gameState !== 'idle' && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">{score}/{total}</Badge>
              {streak >= 3 && <Badge variant="default">Streak: {streak}</Badge>}
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Select value={language} onValueChange={v => setLanguage(v as PhonemeLanguage)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PHONEME_LANGUAGES.map(lang => (
                <SelectItem key={lang.value} value={lang.value}>{lang.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {gameState === 'idle' && (
          <div className="text-center py-6">
            <p className="text-sm text-muted-foreground mb-4">
              Identify the consonant/vowel pattern of each word.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mb-4">
              {PATTERNS.map(p => (
                <Badge key={p.pattern} variant="outline" className="text-xs">
                  {p.pattern}: {p.label}
                </Badge>
              ))}
            </div>
            <Button onClick={startGame}>Start Quiz</Button>
          </div>
        )}

        {gameState === 'playing' && question && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-xs text-muted-foreground mb-1">What pattern does this word follow?</p>
              <p className="text-xl font-mono font-bold">{question.word}</p>
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                {question.phonemes.join('-')}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {question.options.map(opt => {
                const label = PATTERNS.find(p => p.pattern === opt)?.label || opt
                return (
                  <Button
                    key={opt}
                    variant="outline"
                    onClick={() => checkAnswer(opt)}
                    className="h-auto py-3"
                  >
                    <div className="text-center">
                      <p className="font-mono font-bold">{opt}</p>
                      <p className="text-[10px] text-muted-foreground mt-1">{label}</p>
                    </div>
                  </Button>
                )
              })}
            </div>
          </div>
        )}

        {gameState === 'revealed' && question && (
          <div className="space-y-3">
            <div className={`p-4 rounded-lg text-center ${
              question.options.indexOf(question.correctPattern) >= 0
                ? 'bg-green-500/10 border border-green-500/20'
                : ''
            }`}>
              <p className="text-lg font-mono font-bold">{question.correctPattern}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {PATTERNS.find(p => p.pattern === question.correctPattern)?.label}
              </p>
            </div>
            <Button onClick={generateQuestion} className="w-full">
              Next Word
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
