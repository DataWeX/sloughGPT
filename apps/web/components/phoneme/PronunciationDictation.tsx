'use client'

import { useState, useCallback, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import PhonemeSkeleton from './PhonemeSkeleton'

const WORDS: Record<PhonemeLanguage, string[]> = {
  en: ['hello', 'world', 'phone', 'music', 'happy', 'green', 'blue', 'apple', 'tree', 'fish'],
  de: ['hallo', 'welt', 'musik', 'glück', 'buch', 'schule', 'grün', 'rot', 'haus', 'katze'],
  fr: ['bonjour', 'monde', 'musique', 'heure', 'heureux', 'bleu', 'vert', 'pomme', 'arbre', 'poisson'],
  es: ['hola', 'mundo', 'música', 'feliz', 'verde', 'azul', 'manzana', 'árbol', 'pez', 'gato'],
  it: ['ciao', 'mondo', 'musica', 'felice', 'verde', 'blu', 'mela', 'albero', 'pesce', 'gatto'],
  pt: ['olá', 'mundo', 'música', 'feliz', 'verde', 'azul', 'maçã', 'árvore', 'peixe', 'gato'],
}

type GameState = 'idle' | 'playing' | 'revealed'

export default function PronunciationDictation() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [gameState, setGameState] = useState<GameState>('idle')
  const [currentWord, setCurrentWord] = useState('')
  const [userInput, setUserInput] = useState('')
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const pickNewWord = useCallback(() => {
    const words = WORDS[language]
    const word = words[Math.floor(Math.random() * words.length)]
    setCurrentWord(word)
    setUserInput('')
    setGameState('playing')
  }, [language])

  const startGame = useCallback(() => {
    setScore(0)
    setTotal(0)
    pickNewWord()
  }, [pickNewWord])

  const checkAnswer = useCallback(async () => {
    if (!userInput.trim() || !currentWord) return

    const isCorrect = userInput.trim().toLowerCase() === currentWord.toLowerCase()
    setTotal(t => t + 1)
    if (isCorrect) setScore(s => s + 1)
    setGameState('revealed')

    try {
      const result = await phonemeController.encode(currentWord, language)
      addToHistory({
        targetWord: currentWord,
        spokenWord: userInput.trim(),
        targetPhonemes: result.phonemes,
        spokenPhonemes: result.phonemes,
        scores: isCorrect ? [1.0] : [0.3],
        language,
        timestamp: new Date().toISOString(),
      })
    } catch {}

    addToast({
      type: isCorrect ? 'success' : 'warning',
      message: isCorrect ? 'Correct!' : `The word was "${currentWord}"`,
    })
  }, [userInput, currentWord, language, addToHistory, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation Dictation</span>
          {gameState !== 'idle' && (
            <Badge variant="outline">{score}/{total}</Badge>
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
              Listen to the word (from your practice history) and type what you hear.
            </p>
            <Button onClick={startGame}>Start Dictation</Button>
          </div>
        )}

        {gameState === 'playing' && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-xs text-muted-foreground mb-1">Type the word you practiced:</p>
              <p className="text-xs text-muted-foreground">Hint: {currentWord.length} letters</p>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Type the word..."
                value={userInput}
                onChange={e => setUserInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && checkAnswer()}
                autoFocus
                className="font-mono"
              />
              <Button onClick={checkAnswer} disabled={!userInput.trim()}>Check</Button>
            </div>
          </div>
        )}

        {gameState === 'revealed' && (
          <div className="space-y-3">
            <div className={`p-4 rounded-lg text-center ${
              userInput.toLowerCase() === currentWord.toLowerCase()
                ? 'bg-green-500/10 border border-green-500/20'
                : 'bg-destructive/10 border border-destructive/20'
            }`}>
              <p className="text-sm font-medium">
                {userInput.toLowerCase() === currentWord.toLowerCase() ? 'Correct!' : 'Not quite'}
              </p>
              <p className="text-lg font-mono mt-1">{currentWord}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Your answer: {userInput}
              </p>
            </div>
            <Button onClick={pickNewWord} className="w-full">
              <IconRefresh className="w-4 h-4 mr-2" />
              Next Word
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
