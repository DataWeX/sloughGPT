'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { toIPA } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import PhonemeReference from './PhonemeReference'

const PHONEMES = [
  'IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AH', 'AO', 'OW', 'OY', 'UH', 'UW', 'ER', 'AX',
  'P', 'B', 'T', 'D', 'K', 'G', 'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH',
  'M', 'N', 'NG', 'L', 'R', 'W', 'Y', 'CH', 'JH',
]

type GameState = 'idle' | 'typing' | 'result'

interface Challenge {
  phonemes: string[]
  ipa: string
}

export default function PhonemeTypingMode() {
  const [gameState, setGameState] = useState<GameState>('idle')
  const [challenge, setChallenge] = useState<Challenge | null>(null)
  const [userInput, setUserInput] = useState('')
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const [showRef, setShowRef] = useState(false)
  const addToHistory = usePhonemeStore(s => s.addToHistory)
  const addToast = useToastStore(s => s.addToast)

  const generateChallenge = useCallback(() => {
    const count = 3 + Math.floor(Math.random() * 3)
    const phonemes = Array.from({ length: count }, () =>
      PHONEMES[Math.floor(Math.random() * PHONEMES.length)]
    )
    const ipa = toIPA(phonemes).join('')
    setChallenge({ phonemes, ipa })
    setUserInput('')
    setGameState('typing')
  }, [])

  const startGame = useCallback(() => {
    setScore(0)
    setTotal(0)
    generateChallenge()
  }, [generateChallenge])

  const checkAnswer = useCallback(() => {
    if (!challenge || !userInput.trim()) return

    const normalize = (s: string) => s.toUpperCase().replace(/[^A-Z0-9]/g, '')
    const userPhonemes = normalize(userInput).split(/[\s,\-]+/).filter(Boolean)
    const isCorrect = normalize(userInput) === normalize(challenge.phonemes.join(' '))

    setTotal(t => t + 1)
    if (isCorrect) setScore(s => s + 1)
    setGameState('result')

    addToHistory({
      targetWord: challenge.phonemes.join('-'),
      spokenWord: userInput,
      targetPhonemes: challenge.phonemes,
      spokenPhonemes: userPhonemes,
      scores: isCorrect ? [1.0] : [0.3],
      language: 'en',
      timestamp: new Date().toISOString(),
    })

    addToast({
      type: isCorrect ? 'success' : 'warning',
      message: isCorrect ? 'Correct!' : `Expected: ${challenge.phonemes.join(' ')}`,
    })
  }, [challenge, userInput, addToHistory, addToast])

  const insertPhoneme = useCallback((phoneme: string) => {
    setUserInput(prev => prev ? `${prev} ${phoneme}` : phoneme)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phoneme Typing</span>
          {gameState !== 'idle' && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">{score}/{total}</Badge>
              <Button variant="ghost" size="sm" onClick={() => setShowRef(!showRef)}>
                {showRef ? 'Hide' : 'Show'} Reference
              </Button>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {gameState === 'idle' && (
          <div className="text-center py-6">
            <p className="text-sm text-muted-foreground mb-4">
              Type phonemes from the reference panel. Test your knowledge of ARPAbet symbols.
            </p>
            <Button onClick={startGame}>Start Typing</Button>
          </div>
        )}

        {gameState === 'typing' && challenge && (
          <div className="space-y-3">
            <div className="p-4 rounded-lg bg-muted/30 text-center">
              <p className="text-xs text-muted-foreground mb-1">Type these phonemes (space-separated):</p>
              <p className="text-lg font-mono font-bold">{challenge.ipa}</p>
              <p className="text-xs text-muted-foreground mt-1">{challenge.phonemes.length} phonemes</p>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Type phonemes (e.g., DH AH K AE T)"
                value={userInput}
                onChange={e => setUserInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && checkAnswer()}
                autoFocus
                className="font-mono"
              />
              <Button onClick={checkAnswer} disabled={!userInput.trim()}>Check</Button>
            </div>
            <div className="flex flex-wrap gap-1">
              {PHONEMES.map(p => (
                <Button
                  key={p}
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs font-mono"
                  onClick={() => insertPhoneme(p)}
                >
                  {p}
                </Button>
              ))}
            </div>
          </div>
        )}

        {gameState === 'result' && challenge && (
          <div className="space-y-3">
            <div className={`p-4 rounded-lg text-center ${
              userInput.toUpperCase().replace(/[^A-Z0-9]/g, '') ===
              challenge.phonemes.join('').toUpperCase()
                ? 'bg-green-500/10 border border-green-500/20'
                : 'bg-destructive/10 border border-destructive/20'
            }`}>
              <p className="text-sm font-medium">
                {userInput.toUpperCase().replace(/[^A-Z0-9]/g, '') ===
                 challenge.phonemes.join('').toUpperCase()
                  ? 'Correct!' : 'Not quite'}
              </p>
              <p className="text-lg font-mono mt-1">{challenge.phonemes.join(' ')}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Your answer: {userInput}
              </p>
            </div>
            <Button onClick={generateChallenge} className="w-full">
              <IconRefresh className="w-4 h-4 mr-2" />
              Next Challenge
            </Button>
          </div>
        )}

        {showRef && <PhonemeReference />}
      </CardContent>
    </Card>
  )
}
