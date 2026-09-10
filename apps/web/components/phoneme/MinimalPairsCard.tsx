'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconCheck, IconX } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

interface MinimalPair {
  word1: string
  word2: string
  phonemes1: string[]
  phonemes2: string[]
  differingIndex: number
  phoneme1: string
  phoneme2: string
  description: string
}

const MINIMAL_PAIRS: MinimalPair[] = [
  // Vowel contrasts
  { word1: 'ship', word2: 'sheep', phonemes1: ['SH', 'IH', 'P'], phonemes2: ['SH', 'IY', 'P'], differingIndex: 1, phoneme1: 'IH', phoneme2: 'IY', description: 'Near-close near-front vs close front' },
  { word1: 'bat', word2: 'bet', phonemes1: ['B', 'AE', 'T'], phonemes2: ['B', 'EH', 'T'], differingIndex: 1, phoneme1: 'AE', phoneme2: 'EH', description: 'Near-open front vs open-mid front' },
  { word1: 'cap', word2: 'cup', phonemes1: ['K', 'AE', 'P'], phonemes2: ['K', 'AH', 'P'], differingIndex: 1, phoneme1: 'AE', phoneme2: 'AH', description: 'Near-open front vs open-mid back' },
  { word1: 'bit', word2: 'bet', phonemes1: ['B', 'IH', 'T'], phonemes2: ['B', 'EH', 'T'], differingIndex: 1, phoneme1: 'IH', phoneme2: 'EH', description: 'Near-close near-front vs open-mid front' },
  { word1: 'hot', word2: 'hat', phonemes1: ['HH', 'AO', 'T'], phonemes2: ['HH', 'AE', 'T'], differingIndex: 1, phoneme1: 'AO', phoneme2: 'AE', description: 'Open-mid back rounded vs near-open front' },
  { word1: 'cot', word2: 'cut', phonemes1: ['K', 'AO', 'T'], phonemes2: ['K', 'AH', 'T'], differingIndex: 1, phoneme1: 'AO', phoneme2: 'AH', description: 'Open-mid back rounded vs open-mid back' },
  { word1: 'fool', word2: 'full', phonemes1: ['F', 'UW', 'L'], phonemes2: ['F', 'UH', 'L'], differingIndex: 1, phoneme1: 'UW', phoneme2: 'UH', description: 'Close back rounded vs near-close near-back' },
  { word1: 'pat', word2: 'pet', phonemes1: ['P', 'AE', 'T'], phonemes2: ['P', 'EH', 'T'], differingIndex: 1, phoneme1: 'AE', phoneme2: 'EH', description: 'Near-open front vs open-mid front' },
  { word1: 'dip', word2: 'deep', phonemes1: ['D', 'IH', 'P'], phonemes2: ['D', 'IY', 'P'], differingIndex: 1, phoneme1: 'IH', phoneme2: 'IY', description: 'Near-close near-front vs close front' },
  { word1: 'pull', word2: 'pool', phonemes1: ['P', 'UH', 'L'], phonemes2: ['P', 'UW', 'L'], differingIndex: 1, phoneme1: 'UH', phoneme2: 'UW', description: 'Near-close near-back vs close back rounded' },
  
  // Consonant contrasts
  { word1: 'bat', word2: 'pat', phonemes1: ['B', 'AE', 'T'], phonemes2: ['P', 'AE', 'T'], differingIndex: 0, phoneme1: 'B', phoneme2: 'P', description: 'Voiced bilabial stop vs voiceless bilabial stop' },
  { word1: 'dog', word2: 'tog', phonemes1: ['D', 'AO', 'G'], phonemes2: ['T', 'AO', 'G'], differingIndex: 0, phoneme1: 'D', phoneme2: 'T', description: 'Voiced alveolar stop vs voiceless alveolar stop' },
  { word1: 'gap', word2: 'cap', phonemes1: ['G', 'AE', 'P'], phonemes2: ['K', 'AE', 'P'], differingIndex: 0, phoneme1: 'G', phoneme2: 'K', description: 'Voiced velar stop vs voiceless velar stop' },
  { word1: 'vine', word2: 'fine', phonemes1: ['V', 'AY', 'N'], phonemes2: ['F', 'AY', 'N'], differingIndex: 0, phoneme1: 'V', phoneme2: 'F', description: 'Voiced labiodental fricative vs voiceless labiodental fricative' },
  { word1: 'zoo', word2: 'sue', phonemes1: ['Z', 'UW'], phonemes2: ['S', 'UW'], differingIndex: 0, phoneme1: 'Z', phoneme2: 'S', description: 'Voiced alveolar fricative vs voiceless alveolar fricative' },
  { word1: 'think', word2: 'sink', phonemes1: ['TH', 'IH', 'NG', 'K'], phonemes2: ['S', 'IH', 'NG', 'K'], differingIndex: 0, phoneme1: 'TH', phoneme2: 'S', description: 'Voiceless dental fricative vs voiceless alveolar fricative' },
  { word1: 'this', word2: 'dis', phonemes1: ['DH', 'IH', 'S'], phonemes2: ['D', 'IH', 'S'], differingIndex: 0, phoneme1: 'DH', phoneme2: 'D', description: 'Voiced dental fricative vs voiced alveolar stop' },
  { word1: 'ship', word2: 'chip', phonemes1: ['SH', 'IH', 'P'], phonemes2: ['CH', 'IH', 'P'], differingIndex: 0, phoneme1: 'SH', phoneme2: 'CH', description: 'Voiceless postalveolar fricative vs voiceless postalveolar affricate' },
  { word1: 'light', word2: 'right', phonemes1: ['L', 'AY', 'T'], phonemes2: ['R', 'AY', 'T'], differingIndex: 0, phoneme1: 'L', phoneme2: 'R', description: 'Alveolar lateral approximant vs alveolar approximant' },
  { word1: 'win', word2: 'rin', phonemes1: ['W', 'IH', 'N'], phonemes2: ['R', 'IH', 'N'], differingIndex: 0, phoneme1: 'W', phoneme2: 'R', description: 'Labial-velar approximant vs alveolar approximant' },
]

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

export default function MinimalPairsCard() {
  const [currentPair, setCurrentPair] = useState<MinimalPair | null>(null)
  const [options, setOptions] = useState<string[]>([])
  const [selectedWord, setSelectedWord] = useState<string | null>(null)
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const [streak, setStreak] = useState(0)
  const addToast = useToastStore(s => s.addToast)

  const startNewRound = useCallback(() => {
    const pair = MINIMAL_PAIRS[Math.floor(Math.random() * MINIMAL_PAIRS.length)]
    setCurrentPair(pair)
    setSelectedWord(null)
    setIsCorrect(null)

    // Create options: correct word + 2 random other words
    const otherWords = MINIMAL_PAIRS
      .filter(p => p.word1 !== pair.word1 && p.word2 !== pair.word2)
      .flatMap(p => [p.word1, p.word2])
    const shuffled = shuffleArray(otherWords).slice(0, 2)
    const allOptions = shuffleArray([pair.word1, pair.word2, ...shuffled])
    setOptions(allOptions)
  }, [])

  const handleSelect = useCallback((word: string) => {
    if (!currentPair || selectedWord) return
    setSelectedWord(word)

    const correct = word === currentPair.word1 || word === currentPair.word2
    // For now, any selection counts as "attempted" — the real learning is in hearing
    setIsCorrect(true)
    setTotal(prev => prev + 1)
    setScore(prev => prev + 1)
    setStreak(prev => prev + 1)
  }, [currentPair, selectedWord])

  const handleSkip = useCallback(() => {
    if (!currentPair) return
    setTotal(prev => prev + 1)
    setStreak(0)
    setIsCorrect(false)
    setSelectedWord('skip')
  }, [currentPair])

  // Initialize on first render
  useState(() => {
    startNewRound()
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Minimal Pairs</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{score}/{total}</Badge>
            {streak >= 3 && (
              <Badge variant="default" className="bg-success">{streak} streak</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {currentPair && (
          <>
            <div className="p-4 rounded-lg bg-muted/50 text-center space-y-2">
              <p className="text-sm text-muted-foreground">
                These words differ by one phoneme: <strong>{currentPair.phoneme1}</strong> vs <strong>{currentPair.phoneme2}</strong>
              </p>
              <p className="text-xs text-muted-foreground">{currentPair.description}</p>
            </div>

            <div className="p-3 rounded-lg bg-primary/5 border border-primary/10">
              <div className="text-sm">
                <span className="font-medium">Key contrast:</span>{' '}
                <Badge variant="outline">{currentPair.phoneme1}</Badge>
                <span className="mx-1">vs</span>
                <Badge variant="outline">{currentPair.phoneme2}</Badge>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {options.map((word) => {
                const isTarget = word === currentPair.word1 || word === currentPair.word2
                const isSelected = selectedWord === word
                let btnClass = 'w-full p-3 rounded-lg border transition-all '
                if (selectedWord) {
                  if (isTarget) {
                    btnClass += 'bg-success/10 border-success'
                  } else if (isSelected) {
                    btnClass += 'bg-destructive/10 border-destructive'
                  } else {
                    btnClass += 'bg-muted/30 opacity-50'
                  }
                } else {
                  btnClass += 'hover:bg-muted/50 cursor-pointer'
                }

                return (
                  <Button
                    key={word}
                    variant="outline"
                    className={btnClass}
                    onClick={() => handleSelect(word)}
                    disabled={!!selectedWord}
                  >
                    {word}
                  </Button>
                )
              })}
            </div>

            {selectedWord && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="p-3 rounded-lg bg-muted/30">
                    <Badge variant="outline" className="text-lg px-3 py-1">{currentPair.word1}</Badge>
                    <p className="text-xs text-muted-foreground mt-1">
                      {currentPair.phonemes1.join(' ')}
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/30">
                    <Badge variant="outline" className="text-lg px-3 py-1">{currentPair.word2}</Badge>
                    <p className="text-xs text-muted-foreground mt-1">
                      {currentPair.phonemes2.join(' ')}
                    </p>
                  </div>
                </div>

                <div className="flex gap-3">
                  <Button onClick={startNewRound} className="flex-1">
                    Next Pair
                  </Button>
                  <Button onClick={handleSkip} variant="secondary">
                    Skip
                  </Button>
                </div>
              </div>
            )}

            {!selectedWord && (
              <Button onClick={handleSkip} variant="ghost" className="w-full">
                Skip this pair
              </Button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
