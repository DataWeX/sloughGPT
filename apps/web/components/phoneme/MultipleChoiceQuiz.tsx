'use client'

import { useState, useCallback, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconRefresh, IconBolt } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { phonemeController, PHONEME_LANGUAGES, type PhonemeLanguage } from '@/lib/phoneme-controller'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'
import PhonemeSkeleton from './PhonemeSkeleton'

type QuizType = 'phoneme-to-word' | 'word-to-phoneme' | 'ipa-to-phoneme'

interface QuizQuestion {
  type: QuizType
  word?: string
  phonemes?: string[]
  ipa?: string
  options: string[]
  correctAnswer: string
}

const QUIZ_WORDS: { word: string; lang: PhonemeLanguage; phonemes: string[] }[] = [
  { word: 'cat', lang: 'en', phonemes: ['K', 'AE', 'T'] },
  { word: 'dog', lang: 'en', phonemes: ['D', 'AO', 'G'] },
  { word: 'hello', lang: 'en', phonemes: ['HH', 'AH', 'L', 'OW'] },
  { word: 'world', lang: 'en', phonemes: ['W', 'ER', 'L', 'D'] },
  { word: 'think', lang: 'en', phonemes: ['TH', 'IH', 'NG', 'K'] },
  { word: 'church', lang: 'en', phonemes: ['CH', 'ER', 'CH'] },
  { word: 'fish', lang: 'en', phonemes: ['F', 'IH', 'SH'] },
  { word: 'bird', lang: 'en', phonemes: ['B', 'ER', 'D'] },
  { word: 'chat', lang: 'fr', phonemes: ['SH', 'AA'] },
  { word: 'bonjour', lang: 'fr', phonemes: ['B', 'OH', 'N', 'ZH', 'U', 'R'] },
  { word: 'hola', lang: 'es', phonemes: ['OH', 'L', 'AH'] },
  { word: 'gracias', lang: 'es', phonemes: ['G', 'R', 'AA', 'TH', 'I', 'AH', 'S'] },
  { word: 'ciao', lang: 'it', phonemes: ['CH', 'AW'] },
  { word: 'gnocchi', lang: 'it', phonemes: ['NY', 'OH', 'K', 'I'] },
  { word: 'sim', lang: 'pt', phonemes: ['S', 'I', 'M'] },
  { word: 'obrigado', lang: 'pt', phonemes: ['OH', 'B', 'R', 'I', 'G', 'AA', 'D', 'UH'] },
]

function generateQuizQuestion(type: QuizType, lang: PhonemeLanguage): QuizQuestion {
  const filteredWords = QUIZ_WORDS.filter(w => w.lang === lang)
  const word = filteredWords[Math.floor(Math.random() * filteredWords.length)]
  const otherWords = QUIZ_WORDS.filter(w => w.word !== word.word)
  
  if (type === 'phoneme-to-word') {
    const options = [word.word, ...otherWords.slice(0, 3).map(w => w.word)].sort(() => Math.random() - 0.5)
    return {
      type,
      phonemes: word.phonemes,
      options,
      correctAnswer: word.word,
    }
  } else if (type === 'word-to-phoneme') {
    const correctPhonemes = word.phonemes.join(' ')
    const wrongPhonemes = otherWords.slice(0, 3).map(w => w.phonemes.join(' '))
    const options = [correctPhonemes, ...wrongPhonemes].sort(() => Math.random() - 0.5)
    return {
      type,
      word: word.word,
      options,
      correctAnswer: correctPhonemes,
    }
  } else {
    const ipa = word.phonemes.join(' ')
    const options = [ipa, ...otherWords.slice(0, 3).map(w => w.phonemes.join(' '))].sort(() => Math.random() - 0.5)
    return {
      type,
      ipa,
      options,
      correctAnswer: ipa,
    }
  }
}

export default function MultipleChoiceQuiz() {
  const [language, setLanguage] = useState<PhonemeLanguage>('en')
  const [quizType, setQuizType] = useState<QuizType>('phoneme-to-word')
  const [question, setQuestion] = useState<QuizQuestion | null>(null)
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null)
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(false)
  const [score, setScore] = useState(0)
  const [total, setTotal] = useState(0)
  const [streak, setStreak] = useState(0)
  const [bestStreak, setBestStreak] = useState(0)
  const addToast = useToastStore(s => s.addToast)

  const startNewQuestion = useCallback(() => {
    setLoading(true)
    setSelectedAnswer(null)
    setIsCorrect(null)
    setTimeout(() => {
      const q = generateQuizQuestion(quizType, language)
      setQuestion(q)
      setLoading(false)
    }, 300)
  }, [quizType, language])

  useEffect(() => {
    startNewQuestion()
  }, [startNewQuestion])

  const handleAnswer = useCallback((answer: string) => {
    if (!question || selectedAnswer) return
    setSelectedAnswer(answer)
    const correct = answer === question.correctAnswer
    setIsCorrect(correct)
    setTotal(prev => prev + 1)
    
    if (correct) {
      setScore(prev => prev + 1)
      setStreak(prev => {
        const newStreak = prev + 1
        setBestStreak(best => Math.max(best, newStreak))
        if (newStreak >= 3) addToast(`${newStreak} streak!`, 'success')
        return newStreak
      })
    } else {
      setStreak(0)
    }
  }, [question, selectedAnswer, addToast])

  const handleNext = useCallback(() => {
    startNewQuestion()
  }, [startNewQuestion])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Multiple Choice Quiz</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{score}/{total}</Badge>
            {streak >= 3 && (
              <Badge variant="default" className="gap-1 bg-warning text-warning-foreground">
                <IconBolt className="h-3 w-3" />
                {streak}
              </Badge>
            )}
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
          <Select value={quizType} onValueChange={v => setQuizType(v as QuizType)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="phoneme-to-word">Phonemes → Word</SelectItem>
              <SelectItem value="word-to-phoneme">Word → Phonemes</SelectItem>
              <SelectItem value="ipa-to-phoneme">IPA → Phonemes</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={startNewQuestion} variant="secondary">
            <IconRefresh className="mr-2 h-4 w-4" />
            New Question
          </Button>
        </div>

        {loading && <PhonemeSkeleton variant="quiz" />}

        {question && !loading && (
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-muted/50 text-center">
              {question.type === 'phoneme-to-word' && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Which word matches these phonemes?</p>
                  <div className="flex flex-wrap gap-1 justify-center">
                    {question.phonemes?.map((p, i) => (
                      <Badge key={i} variant="outline" className="text-lg px-3 py-1">{p}</Badge>
                    ))}
                  </div>
                </div>
              )}
              {question.type === 'word-to-phoneme' && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Which phoneme sequence matches this word?</p>
                  <p className="text-2xl font-bold">{question.word}</p>
                </div>
              )}
              {question.type === 'ipa-to-phoneme' && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Which phoneme sequence matches this IPA?</p>
                  <p className="text-2xl font-mono">{question.ipa}</p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {question.options.map((option, i) => {
                const isSelected = selectedAnswer === option
                const isCorrectOption = option === question.correctAnswer
                let buttonClass = 'w-full text-left p-3 rounded-lg border transition-colors '
                
                if (selectedAnswer) {
                  if (isCorrectOption) {
                    buttonClass += 'bg-success/10 border-success'
                  } else if (isSelected && !isCorrectOption) {
                    buttonClass += 'bg-destructive/10 border-destructive'
                  } else {
                    buttonClass += 'bg-muted/30 opacity-50'
                  }
                } else {
                  buttonClass += 'hover:bg-muted/50'
                }

                return (
                  <Button
                    key={i}
                    variant="outline"
                    className={buttonClass}
                    onClick={() => handleAnswer(option)}
                    disabled={!!selectedAnswer}
                  >
                    <span className="font-mono">{option}</span>
                  </Button>
                )
              })}
            </div>

            {selectedAnswer && (
              <div className="space-y-3">
                <div className={`p-3 rounded-lg text-center ${isCorrect ? 'bg-success/10' : 'bg-destructive/10'}`}>
                  <p className={`text-lg font-bold ${isCorrect ? 'text-success' : 'text-destructive'}`}>
                    {isCorrect ? 'Correct!' : 'Incorrect'}
                  </p>
                  {!isCorrect && (
                    <p className="text-sm text-muted-foreground">
                      Answer: {question.correctAnswer}
                    </p>
                  )}
                </div>
                <Button onClick={handleNext} className="w-full">
                  Next Question
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
