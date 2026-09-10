'use client'

import { useState, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconCheck, IconX } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'

interface Flashcard {
  word: string
  lang: PhonemeLanguage
  phonemes: string[]
  tip: string
  difficulty: 'easy' | 'medium' | 'hard'
}

const FLASHCARD_BANK: Flashcard[] = [
  { word: 'cat', lang: 'en', phonemes: ['K', 'AE', 'T'], tip: 'The vowel /æ/ is produced with the tongue low and front. Open your mouth wider than for /ɛ/.', difficulty: 'easy' },
  { word: 'church', lang: 'en', phonemes: ['CH', 'ER', 'CH'], tip: 'The /tʃ/ affricate starts with a stop and releases into a fricative. Keep the tongue curled back.', difficulty: 'medium' },
  { word: 'think', lang: 'en', phonemes: ['TH', 'IH', 'NG', 'K'], tip: '/θ/ is a voiceless dental fricative. Place tongue between teeth, do not vibrate vocal cords.', difficulty: 'hard' },
  { word: 'Schadenfreude', lang: 'de', phonemes: ['SH', 'AA', 'D', 'AH', 'N', 'F', 'R', 'OY', 'D', 'AH'], tip: 'The /ʃ/ is postalveolar. The diphthong /ɔʏ/ rounds then spreads. German /r/ is uvular.', difficulty: 'hard' },
  { word: 'Wanderlust', lang: 'de', phonemes: ['V', 'AA', 'N', 'D', 'ER', 'L', 'U', 'S', 'T'], tip: 'German /v/ is a labiodental fricative, like English "v" not "w". The final cluster is devoiced.', difficulty: 'medium' },
  { word: 'chat', lang: 'fr', phonemes: ['SH', 'AA'], tip: 'French "ch" is always /ʃ/. The vowel /a/ is open front. French has no final consonant stress.', difficulty: 'easy' },
  { word: 'bonjour', lang: 'fr', phonemes: ['B', 'OH', 'N', 'ZH', 'U', 'R'], tip: 'The nasal /ɔ̃/ is produced by lowering the velum. /ʁ/ is uvular, produced in the back.', difficulty: 'medium' },
  { word: 'perro', lang: 'es', phonemes: ['P', 'EH', 'R', 'R', 'OH'], tip: 'Spanish trilled /r/ uses the tongue tip against the alveolar ridge. The geminate /rr/ is stronger.', difficulty: 'medium' },
  { word: 'gracias', lang: 'es', phonemes: ['G', 'R', 'AA', 'TH', 'I', 'AH', 'S'], tip: 'Spanish /θ/ (in Castilian) is dental. The /a/ is open and unaspirated.', difficulty: 'medium' },
  { word: 'ciao', lang: 'it', phonemes: ['CH', 'AW'], tip: 'Italian "ch" before "i" or "e" is /k/. The diphthong glides from open to close back vowel.', difficulty: 'easy' },
  { word: 'gnocchi', lang: 'it', phonemes: ['NY', 'OH', 'K', 'I'], tip: 'Italian "gn" is a palatal nasal /ɲ/. The tongue presses against the hard palate.', difficulty: 'hard' },
  { word: 'saudade', lang: 'pt', phonemes: ['S', 'AW', 'D', 'AA', 'D', 'UH'], tip: 'Portuguese nasal /ãw/ blends a nasal diphthong. The final /d/ is often dental and soft.', difficulty: 'hard' },
]

const FLASHCARDS_BY_LANG: Record<string, Flashcard[]> = {}
for (const card of FLASHCARD_BANK) {
  if (!FLASHCARDS_BY_LANG[card.lang]) FLASHCARDS_BY_LANG[card.lang] = []
  FLASHCARDS_BY_LANG[card.lang].push(card)
}

function getCardKey(card: Flashcard): string {
  return `${card.lang}:${card.word}`
}

export default function FlashcardCard() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isFlipped, setIsFlipped] = useState(false)
  const [studyMode, setStudyMode] = useState<'all' | 'due' | 'new'>('all')

  const flashcardSRS = usePhonemeStore(s => s.flashcardSRS)
  const updateFlashcardSRS = usePhonemeStore(s => s.updateFlashcardSRS)
  const resetFlashcardSRS = usePhonemeStore(s => s.resetFlashcardSRS)

  const allCards = useMemo(() => {
    return FLASHCARDS_BY_LANG[language] || FLASHCARD_BANK.filter(c => c.lang === language)
  }, [language])

  const cards = useMemo(() => {
    if (studyMode === 'all') return allCards
    const now = Date.now()
    if (studyMode === 'due') {
      return allCards.filter(card => {
        const srs = flashcardSRS[getCardKey(card)]
        return !srs || srs.nextReview <= now
      })
    }
    return allCards.filter(card => !flashcardSRS[getCardKey(card)])
  }, [allCards, studyMode, flashcardSRS])

  const currentCard = cards[currentIndex]

  const handleFlip = useCallback(() => {
    setIsFlipped(prev => !prev)
  }, [])

  const handleKnow = useCallback(() => {
    if (!currentCard) return
    updateFlashcardSRS(getCardKey(currentCard), true)
    setIsFlipped(false)
    setCurrentIndex(prev => (prev + 1) % cards.length)
  }, [currentCard, updateFlashcardSRS, cards.length])

  const handleDontKnow = useCallback(() => {
    if (!currentCard) return
    updateFlashcardSRS(getCardKey(currentCard), false)
    setIsFlipped(false)
    setCurrentIndex(prev => (prev + 1) % cards.length)
  }, [currentCard, updateFlashcardSRS, cards.length])

  const handleReset = useCallback(() => {
    resetFlashcardSRS()
    setCurrentIndex(0)
    setIsFlipped(false)
  }, [resetFlashcardSRS])

  const handleLanguageChange = useCallback((lang: PhonemeLanguage) => {
    setLanguage(lang)
    setCurrentIndex(0)
    setIsFlipped(false)
  }, [])

  const dueCount = useMemo(() => {
    const now = Date.now()
    return allCards.filter(card => {
      const srs = flashcardSRS[getCardKey(card)]
      return !srs || srs.nextReview <= now
    }).length
  }, [allCards, flashcardSRS])

  const masteredCount = useMemo(() => {
    return allCards.filter(card => {
      const srs = flashcardSRS[getCardKey(card)]
      return srs && srs.interval >= 7
    }).length
  }, [allCards, flashcardSRS])

  if (!currentCard) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          {studyMode === 'due' && dueCount === 0
            ? 'No cards due for review. Great job!'
            : studyMode === 'new'
            ? 'All cards have been studied. Switch to "All" or "Due" mode.'
            : 'No flashcards available for this language yet.'}
        </CardContent>
      </Card>
    )
  }

  const cardKey = getCardKey(currentCard)
  const cardSRS = flashcardSRS[cardKey]
  const nextReviewText = cardSRS
    ? cardSRS.nextReview <= Date.now()
      ? 'Now'
      : `${Math.ceil((cardSRS.nextReview - Date.now()) / 86400000)}d`
    : 'New'

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Pronunciation Flashcards</span>
            <div className="flex items-center gap-2">
              <Badge variant="default" className="bg-success text-success-foreground">{masteredCount} mastered</Badge>
              <Badge variant="secondary">{dueCount} due</Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Select value={language} onValueChange={v => handleLanguageChange(v as PhonemeLanguage)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PHONEME_LANGUAGES.map(l => (
                  <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={studyMode} onValueChange={v => { setStudyMode(v as 'all' | 'due' | 'new'); setCurrentIndex(0); setIsFlipped(false) }}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Cards</SelectItem>
                <SelectItem value="due">Due Only</SelectItem>
                <SelectItem value="new">New Only</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="secondary" size="sm" onClick={handleReset}>Reset SRS</Button>
          </div>

          <div className="flex justify-center gap-2 text-xs text-muted-foreground">
            {cards.slice(0, 20).map((card, i) => {
              const srs = flashcardSRS[getCardKey(card)]
              const isMastered = srs && srs.interval >= 7
              const isDue = !srs || srs.nextReview <= Date.now()
              return (
                <button
                  key={i}
                  onClick={() => { setCurrentIndex(i); setIsFlipped(false) }}
                  className={`w-3 h-3 rounded-full transition-colors ${
                    i === currentIndex ? 'bg-primary ring-2 ring-primary/30' :
                    isMastered ? 'bg-success' :
                    isDue ? 'bg-warning' :
                    'bg-muted'
                  }`}
                />
              )
            })}
            {cards.length > 20 && <span className="text-xs">+{cards.length - 20}</span>}
          </div>

          <div
            className="relative w-full h-64 cursor-pointer perspective-1000"
            onClick={handleFlip}
          >
            <div className={`absolute inset-0 transition-transform duration-500 preserve-3d ${isFlipped ? 'rotate-y-180' : ''}`}>
              <div className="absolute inset-0 backface-hidden rounded-xl border bg-card p-6 flex flex-col items-center justify-center space-y-3">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">{currentCard.difficulty}</Badge>
                  {cardSRS && (
                    <Badge variant="outline" className="text-xs">
                      {cardSRS.interval}d interval
                    </Badge>
                  )}
                </div>
                <h3 className="text-3xl font-bold">{currentCard.word}</h3>
                <p className="text-sm text-muted-foreground">{currentCard.lang.toUpperCase()}</p>
                <p className="text-xs text-muted-foreground">Click to reveal · Next: {nextReviewText}</p>
              </div>
              <div className="absolute inset-0 backface-hidden rotate-y-180 rounded-xl border bg-card p-6 flex flex-col items-center justify-center space-y-3">
                <h3 className="text-2xl font-bold">{currentCard.word}</h3>
                <div className="flex flex-wrap gap-1 justify-center">
                  {currentCard.phonemes.map((p, i) => (
                    <Badge key={i} variant="outline">{p}</Badge>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground text-center">{currentCard.tip}</p>
              </div>
            </div>
          </div>

          <div className="flex justify-center gap-3">
            <Button variant="destructive" size="lg" onClick={handleDontKnow}>
              <IconX className="mr-2 h-5 w-5" />
              Don&apos;t Know
            </Button>
            <Button variant="default" size="lg" onClick={handleKnow}>
              <IconCheck className="mr-2 h-5 w-5" />
              Know It
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
