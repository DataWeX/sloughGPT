'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, toIPA, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'

const WORDS: Record<PhonemeLanguage, string[]> = {
  en: ['hello', 'world', 'phone', 'music', 'happy', 'green', 'blue', 'apple', 'tree', 'fish', 'cat', 'dog', 'run', 'jump', 'swim', 'think', 'three', 'this', 'that', 'ship'],
  de: ['hallo', 'welt', 'musik', 'glück', 'buch', 'schule', 'grün', 'rot', 'haus', 'katze', 'hund', 'laufen', 'springen', 'schwimmen', 'essen', 'ich', 'nicht', 'durch', 'über', 'schon'],
  fr: ['bonjour', 'monde', 'musique', 'heure', 'heureux', 'bleu', 'vert', 'pomme', 'arbre', 'poisson', 'chat', 'chien', 'courir', 'sauter', 'nager', 'je', 'ne', 'pas', 'bien', 'tout'],
  es: ['hola', 'mundo', 'música', 'feliz', 'verde', 'azul', 'manzana', 'árbol', 'pez', 'gato', 'perro', 'correr', 'saltar', 'nadar', 'comer', 'yo', 'no', 'bien', 'muy', 'todo'],
  it: ['ciao', 'mondo', 'musica', 'felice', 'verde', 'blu', 'mela', 'albero', 'pesce', 'gatto', 'cane', 'correre', 'saltare', 'nuotare', 'mangiare', 'io', 'non', 'bene', 'molto', 'tutto'],
  pt: ['olá', 'mundo', 'música', 'feliz', 'verde', 'azul', 'maçã', 'árvore', 'peixe', 'gato', 'cachorro', 'correr', 'pular', 'nadar', 'comer', 'eu', 'não', 'bem', 'muito', 'tudo'],
}

type QuizMode = 'reverse' | 'partial' | 'missing' | 'sequence'
type GameState = 'idle' | 'playing' | 'result'

interface QuizQuestion {
  word: string
  phonemes: string[]
  ipa: string
  mode: QuizMode
  prompt: string
  options: string[]
  correctAnswer: string
}

function shuffleArray<T>(arr: T[]): T[] {
  return [...arr].sort(() => Math.random() - 0.5)
}

export default function PhonemeQuizAdvanced() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [mode, setMode] = useState<QuizMode>('reverse')
  const [gameState, setGameState] = useState<GameState>('idle')
  const [question, setQuestion] = useState<QuizQuestion | null>(null)
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const [streak, setStreak] = useState(0)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const generateQuestion = useCallback(async () => {
    const words = WORDS[language]
    const word = words[Math.floor(Math.random() * words.length)]

    try {
      const result = await phonemeController.encode(word, language)
      const phonemes = result.phonemes
      const ipa = toIPA(phonemes).join('')
      let prompt = '', correctAnswer = '', options: string[] = []

      switch (mode) {
        case 'reverse': {
          correctAnswer = word
          const distractors = words.filter(w => w !== word).slice(0, 3)
          options = shuffleArray([word, ...distractors])
          prompt = `Which word has these phonemes: ${phonemes.join('-')}?`
          break
        }
        case 'partial': {
          const hiddenIdx = Math.floor(Math.random() * phonemes.length)
          correctAnswer = phonemes[hiddenIdx]
          const phonemeSet = new Set(phonemes)
          const extra = ['P', 'B', 'T', 'D', 'K', 'G', 'S', 'Z', 'M', 'N', 'L', 'R']
            .filter(p => !phonemeSet.has(p))
          options = shuffleArray([correctAnswer, ...extra.slice(0, 3)])
          const masked = phonemes.map((p, i) => i === hiddenIdx ? '___' : p).join('-')
          prompt = `Fill in the missing phoneme: ${masked}`
          break
        }
        case 'missing': {
          correctAnswer = phonemes.length.toString()
          options = shuffleArray([
            phonemes.length.toString(),
            (phonemes.length - 1).toString(),
            (phonemes.length + 1).toString(),
            (phonemes.length + 2).toString(),
          ].filter((v, i, a) => a.indexOf(v) === i).slice(0, 4))
          prompt = `How many phonemes in "${word}"? (${ipa})`
          break
        }
        case 'sequence': {
          const shuffled = shuffleArray(phonemes)
          correctAnswer = phonemes.join('-')
          const correct = phonemes.join('-')
          const wrong1 = shuffleArray(phonemes).join('-')
          const wrong2 = shuffleArray(phonemes).join('-')
          options = shuffleArray([correct, wrong1, wrong2].filter((v, i, a) => a.indexOf(v) === i).slice(0, 3))
          prompt = `Put the phonemes in order for "${word}":`
          break
        }
      }

      setQuestion({ word, phonemes, ipa, mode, prompt, options, correctAnswer })
      setGameState('playing')
    } catch {
      addToast('Failed to generate question', 'error')
    }
  }, [language, mode, addToast])

  const startGame = useCallback(() => {
    setScore(0)
    setTotal(0)
    setStreak(0)
    generateQuestion()
  }, [generateQuestion])

  const checkAnswer = useCallback((selected: string) => {
    if (!question) return
    const isCorrect = selected === question.correctAnswer
    setTotal(t => t + 1)
    if (isCorrect) { setScore(s => s + 1); setStreak(s => s + 1) } else { setStreak(0) }
    setGameState('result')

    addToHistory({
      target: question.word,
      spoken: question.word,
      targetPhonemes: question.phonemes,
      spokenPhonemes: question.phonemes,
      score: isCorrect ? 1.0 : 0.0,
      language,
    })

    addToast(
      isCorrect ? `Correct! ${question.word}` : `Answer: ${question.correctAnswer}`,
      isCorrect ? 'success' : 'error',
    )
  }, [question, language, addToHistory, addToast])

  const MODES: { value: QuizMode; label: string; desc: string }[] = [
    { value: 'reverse', label: 'Reverse', desc: 'Phonemes → Word' },
    { value: 'partial', label: 'Fill Gap', desc: 'Missing phoneme' },
    { value: 'missing', label: 'Count', desc: 'How many phonemes?' },
    { value: 'sequence', label: 'Sequence', desc: 'Order phonemes' },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Advanced Quiz</span>
          {gameState !== 'idle' && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">{score}/{total}</Badge>
              {streak >= 3 && <Badge variant="default">🔥 {streak}</Badge>}
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
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
          <Select value={mode} onValueChange={v => setMode(v as QuizMode)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODES.map(m => (
                <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {gameState === 'idle' && (
          <div className="text-center py-6">
            <p className="text-sm text-muted-foreground mb-4">
              Harder quiz modes: reverse mapping, gap filling, counting, and sequencing.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mb-4">
              {MODES.map(m => (
                <Badge key={m.value} variant="outline" className="text-xs">
                  {m.label}: {m.desc}
                </Badge>
              ))}
            </div>
            <Button onClick={startGame}>Start Quiz</Button>
          </div>
        )}

        {gameState === 'playing' && question && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-sm text-muted-foreground mb-2">{question.prompt}</p>
              {question.mode === 'sequence' && (
                <p className="text-xs text-muted-foreground font-mono">
                  {question.ipa}
                </p>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {question.options.map(opt => (
                <Button
                  key={opt}
                  variant="outline"
                  onClick={() => checkAnswer(opt)}
                  className="h-auto py-3 font-mono"
                >
                  {opt}
                </Button>
              ))}
            </div>
          </div>
        )}

        {gameState === 'result' && question && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-sm text-muted-foreground mb-1">Answer</p>
              <p className="text-lg font-mono font-bold">{question.correctAnswer}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {question.word} ({question.ipa})
              </p>
            </div>
            <Button onClick={generateQuestion} className="w-full">
              Next Question
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
