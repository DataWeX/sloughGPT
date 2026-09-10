'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'

const WORDS: Record<PhonemeLanguage, string[]> = {
  en: ['hello', 'world', 'phone', 'music', 'happy', 'green', 'blue', 'apple', 'tree', 'fish', 'cat', 'dog', 'run', 'jump', 'swim'],
  de: ['hallo', 'welt', 'musik', 'glück', 'buch', 'schule', 'grün', 'rot', 'haus', 'katze', 'hund', 'laufen', 'springen', 'schwimmen', 'essen'],
  fr: ['bonjour', 'monde', 'musique', 'heure', 'heureux', 'bleu', 'vert', 'pomme', 'arbre', 'poisson', 'chat', 'chien', 'courir', 'sauter', 'nager'],
  es: ['hola', 'mundo', 'música', 'feliz', 'verde', 'azul', 'manzana', 'árbol', 'pez', 'gato', 'perro', 'correr', 'saltar', 'nadar', 'comer'],
  it: ['ciao', 'mondo', 'musica', 'felice', 'verde', 'blu', 'mela', 'albero', 'pesce', 'gatto', 'cane', 'correre', 'saltare', 'nuotare', 'mangiare'],
  pt: ['olá', 'mundo', 'música', 'feliz', 'verde', 'azul', 'maçã', 'árvore', 'peixe', 'gato', 'cachorro', 'correr', 'pular', 'nadar', 'comer'],
}

type GameState = 'idle' | 'playing' | 'finished'

interface RoundResult {
  word: string
  userInput: string
  isCorrect: boolean
  timeMs: number
  score: number
}

const ROUND_TIME = 10000
const TOTAL_ROUNDS = 10

export default function PronunciationGame() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [gameState, setGameState] = useState<GameState>('idle')
  const [currentWord, setCurrentWord] = useState('')
  const [userInput, setUserInput] = useState('')
  const [timeLeft, setTimeLeft] = useState(ROUND_TIME)
  const [round, setRound] = useState(0)
  const [results, setResults] = useState<RoundResult[]>([])
  const [roundStartTime, setRoundStartTime] = useState(0)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => clearTimer()
  }, [clearTimer])

  const pickNewWord = useCallback(() => {
    const words = WORDS[language]
    const word = words[Math.floor(Math.random() * words.length)]
    setCurrentWord(word)
    setUserInput('')
    setTimeLeft(ROUND_TIME)
    setRoundStartTime(Date.now())
  }, [language])

  const startGame = useCallback(() => {
    setGameState('playing')
    setRound(1)
    setResults([])
    pickNewWord()

    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 100) {
          clearTimer()
          return 0
        }
        return prev - 100
      })
    }, 100)
  }, [pickNewWord, clearTimer])

  const submitAnswer = useCallback(async () => {
    if (!currentWord || gameState !== 'playing') return

    clearTimer()
    const timeMs = Date.now() - roundStartTime
    const isCorrect = userInput.trim().toLowerCase() === currentWord.toLowerCase()
    const timeBonus = Math.max(0, 1 - timeMs / ROUND_TIME)
    const score = isCorrect ? Math.round(100 * (0.5 + 0.5 * timeBonus)) : 0

    const result: RoundResult = {
      word: currentWord,
      userInput: userInput.trim(),
      isCorrect,
      timeMs,
      score,
    }
    setResults(prev => [...prev, result])

    try {
      const encoded = await phonemeController.encode(currentWord, language)
      addToHistory({
        target: currentWord,
        spoken: userInput.trim(),
        targetPhonemes: encoded.phonemes,
        spokenPhonemes: encoded.phonemes,
        score: isCorrect ? 1.0 : 0.0,
        language,
      })
    } catch {}

    if (round >= TOTAL_ROUNDS) {
      setGameState('finished')
      return
    }

    setRound(r => r + 1)
    pickNewWord()
  }, [currentWord, userInput, gameState, round, roundStartTime, language, addToHistory, clearTimer, pickNewWord])

  useEffect(() => {
    if (timeLeft === 0 && gameState === 'playing') {
      submitAnswer()
    }
  }, [timeLeft, gameState, submitAnswer])

  const totalScore = results.reduce((sum, r) => sum + r.score, 0)
  const maxScore = results.length * 100
  const correctCount = results.filter(r => r.isCorrect).length
  const avgTime = results.length > 0 ? results.reduce((sum, r) => sum + r.timeMs, 0) / results.length : 0

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Game</span>
          {gameState === 'playing' && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">Round {round}/{TOTAL_ROUNDS}</Badge>
              <Badge variant={timeLeft < 3000 ? 'destructive' : 'outline'}>
                {(timeLeft / 1000).toFixed(1)}s
              </Badge>
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
              Type each word as fast as you can! {TOTAL_ROUNDS} rounds, {ROUND_TIME / 1000}s each.
            </p>
            <Button onClick={startGame}>Start Game</Button>
          </div>
        )}

        {gameState === 'playing' && (
          <div className="space-y-3">
            <div className="h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-all duration-100"
                style={{ width: `${(timeLeft / ROUND_TIME) * 100}%` }}
              />
            </div>
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-2xl font-mono font-bold">{currentWord}</p>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Type the word..."
                value={userInput}
                onChange={e => setUserInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && submitAnswer()}
                autoFocus
                className="font-mono"
              />
              <Button onClick={submitAnswer} disabled={!userInput.trim()}>Submit</Button>
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Score: {totalScore}</span>
              <span>{correctCount}/{results.length} correct</span>
            </div>
          </div>
        )}

        {gameState === 'finished' && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center space-y-2">
              <p className="text-lg font-bold">Game Over!</p>
              <p className="text-3xl font-bold text-primary">{totalScore}/{maxScore}</p>
              <p className="text-sm text-muted-foreground">
                {correctCount}/{TOTAL_ROUNDS} correct | Avg time: {(avgTime / 1000).toFixed(1)}s
              </p>
            </div>
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {results.map((r, i) => (
                <div key={i} className={`flex items-center gap-2 p-1.5 rounded text-xs ${r.isCorrect ? 'bg-green-500/5' : 'bg-destructive/5'}`}>
                  <span className="w-4 text-muted-foreground">{i + 1}</span>
                  <span className="font-mono min-w-[60px]">{r.word}</span>
                  <span className="text-muted-foreground truncate flex-1">{r.userInput}</span>
                  <span className="text-muted-foreground">{(r.timeMs / 1000).toFixed(1)}s</span>
                  <span className={r.isCorrect ? 'text-green-600' : 'text-destructive'}>
                    {r.isCorrect ? `+${r.score}` : 'Miss'}
                  </span>
                </div>
              ))}
            </div>
            <Button onClick={startGame} className="w-full">
              <IconRefresh className="w-4 h-4 mr-2" />
              Play Again
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
