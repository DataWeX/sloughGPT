import { useState, useCallback, useEffect } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Tabs, TabsList, TabsTrigger, TabsContent, Kbd } from '@sloughgpt/strui'
import EncodingCard from '@/components/phoneme/EncodingCard'
import ScoringCard from '@/components/phoneme/ScoringCard'
import ComparisonCard from '@/components/phoneme/ComparisonCard'
import PracticeCard from '@/components/phoneme/PracticeCard'
import BatchCard from '@/components/phoneme/BatchCard'
import DetectLanguageCard from '@/components/phoneme/DetectLanguageCard'
import FlashcardCard from '@/components/phoneme/FlashcardCard'
import ChallengeCard from '@/components/phoneme/ChallengeCard'
import WordOfDayCard from '@/components/phoneme/WordOfDayCard'
import QuizCard from '@/components/phoneme/QuizCard'
import MultipleChoiceQuiz from '@/components/phoneme/MultipleChoiceQuiz'
import SynthesisCard from '@/components/phoneme/SynthesisCard'
import HistoryCard from '@/components/phoneme/HistoryCard'
import ProgressSummary from '@/components/phoneme/ProgressSummary'
import PhonemeDifficulty from '@/components/phoneme/PhonemeDifficulty'
import PronunciationPatterns from '@/components/phoneme/PronunciationPatterns'
import PronunciationJournal from '@/components/phoneme/PronunciationJournal'
import MinimalPairsCard from '@/components/phoneme/MinimalPairsCard'
import RhymeFinder from '@/components/phoneme/RhymeFinder'
import PhonemeFrequency from '@/components/phoneme/PhonemeFrequency'
import PhonemeClusters from '@/components/phoneme/PhonemeClusters'
import PronunciationCoach from '@/components/phoneme/PronunciationCoach'
import CustomWordList from '@/components/phoneme/CustomWordList'
import WordFamilies from '@/components/phoneme/WordFamilies'
import ProgressExport from '@/components/phoneme/ProgressExport'
import Achievements from '@/components/phoneme/Achievements'
import PronunciationTips from '@/components/phoneme/PronunciationTips'
import DailyGoal from '@/components/phoneme/DailyGoal'
import ProficiencyTracker from '@/components/phoneme/ProficiencyTracker'
import PronunciationDictation from '@/components/phoneme/PronunciationDictation'
import PhonemePatternQuiz from '@/components/phoneme/PhonemePatternQuiz'
import WordDifficultyRanker from '@/components/phoneme/WordDifficultyRanker'
import PhonemeTypingMode from '@/components/phoneme/PhonemeTypingMode'
import PronunciationSpeed from '@/components/phoneme/PronunciationSpeed'
import PronunciationComparison from '@/components/phoneme/PronunciationComparison'
import PronunciationHeatmap from '@/components/phoneme/PronunciationHeatmap'
import ConfusionMatrix from '@/components/phoneme/ConfusionMatrix'
import PracticeCalendar from '@/components/phoneme/PracticeCalendar'
import PronunciationGame from '@/components/phoneme/PronunciationGame'
import PhonemeNetwork from '@/components/phoneme/PhonemeNetwork'
import PronunciationRecording from '@/components/phoneme/PronunciationRecording'
import PhonemeQuizAdvanced from '@/components/phoneme/PhonemeQuizAdvanced'
import PhonemeReference from '@/components/phoneme/PhonemeReference'
import TabErrorBoundary from '@/components/phoneme/TabErrorBoundary'
import { usePhonemeShortcuts } from '@/hooks/usePhonemeShortcuts'
import { usePhonemeStore } from '@/lib/phoneme-store'

const TAB_STORAGE_KEY = 'sloughgpt-phoneme-tab'

function loadTab(): string {
  if (typeof window === 'undefined') return 'encode'
  try {
    const stored = localStorage.getItem(TAB_STORAGE_KEY)
    if (stored && TABS.some(t => t.value === stored)) return stored
  } catch {}
  return 'encode'
}

function saveTab(tab: string) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(TAB_STORAGE_KEY, tab)
  } catch {}
}

const TABS = [
  { value: 'encode', label: 'Encode', shortcut: '1' },
  { value: 'score', label: 'Score', shortcut: '2' },
  { value: 'compare', label: 'Compare', shortcut: '3' },
  { value: 'practice', label: 'Practice', shortcut: '4' },
  { value: 'batch', label: 'Batch', shortcut: '5' },
  { value: 'detect', label: 'Detect', shortcut: '6' },
  { value: 'flashcards', label: 'Flashcards', shortcut: '7' },
  { value: 'challenge', label: 'Challenge', shortcut: '8' },
  { value: 'quiz', label: 'Quiz', shortcut: '9' },
  { value: 'multi', label: 'Multi Quiz', shortcut: '0' },
  { value: 'synthesize', label: 'Synthesize', shortcut: '-' },
  { value: 'history', label: 'History', shortcut: '=' },
] as const

export default function PhonemePage() {
  const [activeTab, setActiveTab] = useState('encode')
  const triggerRandomWord = usePhonemeStore(s => s.triggerRandomWord)
  const clearHistory = usePhonemeStore(s => s.clearHistory)

  useEffect(() => {
    setActiveTab(loadTab())
  }, [])

  const goToTab = useCallback((value: string) => {
    setActiveTab(value)
    saveTab(value)
  }, [])

  const handleSubmit = useCallback(() => {
    const event = new CustomEvent('phoneme-submit', { detail: { tab: activeTab } })
    window.dispatchEvent(event)
  }, [activeTab])

  usePhonemeShortcuts({
    onEncode: () => goToTab('encode'),
    onScore: () => goToTab('score'),
    onCompare: () => goToTab('compare'),
    onPractice: () => goToTab('practice'),
    onBatch: () => goToTab('batch'),
    onDetect: () => goToTab('detect'),
    onQuiz: () => goToTab('quiz'),
    onSynthesize: () => goToTab('synthesize'),
    onHistory: () => goToTab('history'),
    onRandomWord: () => {
      goToTab('practice')
      triggerRandomWord()
    },
    onClear: () => {
      clearHistory()
    },
    onSubmit: handleSubmit,
  })

  return (
    <PageContainer
      title="Phoneme Encoder"
      subtitle="Multi-language phoneme encoding, pronunciation scoring, and TTS"
      headerRight={
        <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
          <Kbd>Ctrl+1-0</Kbd>
          <span>tabs</span>
          <span className="text-border">|</span>
          <Kbd>Ctrl+Enter</Kbd>
          <span>submit</span>
          <span className="text-border">|</span>
          <Kbd>Ctrl+Shift+R</Kbd>
          <span>random</span>
          <span className="text-border">|</span>
          <Kbd>Ctrl+Shift+X</Kbd>
          <span>clear</span>
        </div>
      }
    >
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          {TABS.map(tab => (
            <TabsTrigger key={tab.value} value={tab.value} className="relative">
              {tab.label}
              <Kbd className="ml-1.5 hidden lg:inline-flex">{tab.shortcut}</Kbd>
            </TabsTrigger>
          ))}
        </TabsList>

        <div className="animate-in fade-in duration-200" key={activeTab}>
          <TabsContent value="encode">
            <TabErrorBoundary tabName="Encode"><EncodingCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="score">
            <TabErrorBoundary tabName="Score"><ScoringCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="compare">
            <TabErrorBoundary tabName="Compare"><ComparisonCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="practice">
            <div className="space-y-4">
              <TabErrorBoundary tabName="Practice"><PracticeCard /></TabErrorBoundary>
              <TabErrorBoundary tabName="Daily Goal"><DailyGoal /></TabErrorBoundary>
              <TabErrorBoundary tabName="Pronunciation Game"><PronunciationGame /></TabErrorBoundary>
              <TabErrorBoundary tabName="Recording"><PronunciationRecording /></TabErrorBoundary>
              <TabErrorBoundary tabName="Word of the Day"><WordOfDayCard /></TabErrorBoundary>
              <TabErrorBoundary tabName="Minimal Pairs"><MinimalPairsCard /></TabErrorBoundary>
              <TabErrorBoundary tabName="Rhyme Finder"><RhymeFinder /></TabErrorBoundary>
              <TabErrorBoundary tabName="Dictation"><PronunciationDictation /></TabErrorBoundary>
              <TabErrorBoundary tabName="Pattern Quiz"><PhonemePatternQuiz /></TabErrorBoundary>
              <TabErrorBoundary tabName="Advanced Quiz"><PhonemeQuizAdvanced /></TabErrorBoundary>
              <TabErrorBoundary tabName="Phoneme Typing"><PhonemeTypingMode /></TabErrorBoundary>
              <TabErrorBoundary tabName="Pronunciation Coach"><PronunciationCoach /></TabErrorBoundary>
              <TabErrorBoundary tabName="Pronunciation Tips"><PronunciationTips /></TabErrorBoundary>
              <TabErrorBoundary tabName="Custom Words"><CustomWordList /></TabErrorBoundary>
              <TabErrorBoundary tabName="Word Families"><WordFamilies /></TabErrorBoundary>
            </div>
          </TabsContent>

          <TabsContent value="batch">
            <TabErrorBoundary tabName="Batch"><BatchCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="detect">
            <TabErrorBoundary tabName="Detect"><DetectLanguageCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="flashcards">
            <TabErrorBoundary tabName="Flashcards"><FlashcardCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="challenge">
            <TabErrorBoundary tabName="Challenge"><ChallengeCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="quiz">
            <TabErrorBoundary tabName="Quiz"><QuizCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="multi">
            <TabErrorBoundary tabName="Multi Quiz"><MultipleChoiceQuiz /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="synthesize">
            <TabErrorBoundary tabName="Synthesize"><SynthesisCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="history">
            <div className="space-y-4">
              <TabErrorBoundary tabName="Practice Calendar"><PracticeCalendar /></TabErrorBoundary>
              <TabErrorBoundary tabName="Progress"><ProgressSummary /></TabErrorBoundary>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <TabErrorBoundary tabName="Phoneme Difficulty"><PhonemeDifficulty /></TabErrorBoundary>
                <TabErrorBoundary tabName="Phoneme Frequency"><PhonemeFrequency /></TabErrorBoundary>
              </div>
              <TabErrorBoundary tabName="Language Proficiency"><ProficiencyTracker /></TabErrorBoundary>
              <TabErrorBoundary tabName="Word Difficulty"><WordDifficultyRanker /></TabErrorBoundary>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <TabErrorBoundary tabName="Pronunciation Heatmap"><PronunciationHeatmap /></TabErrorBoundary>
                <TabErrorBoundary tabName="Confusion Patterns"><ConfusionMatrix /></TabErrorBoundary>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <TabErrorBoundary tabName="Practice Speed"><PronunciationSpeed /></TabErrorBoundary>
                <TabErrorBoundary tabName="Word Comparison"><PronunciationComparison /></TabErrorBoundary>
              </div>
              <TabErrorBoundary tabName="Phoneme Clusters"><PhonemeClusters /></TabErrorBoundary>
              <TabErrorBoundary tabName="Phoneme Network"><PhonemeNetwork /></TabErrorBoundary>
              <TabErrorBoundary tabName="Word Families"><WordFamilies /></TabErrorBoundary>
              <TabErrorBoundary tabName="Pronunciation Patterns"><PronunciationPatterns /></TabErrorBoundary>
              <TabErrorBoundary tabName="Journal"><PronunciationJournal /></TabErrorBoundary>
              <TabErrorBoundary tabName="History"><HistoryCard /></TabErrorBoundary>
              <TabErrorBoundary tabName="Export / Import"><ProgressExport /></TabErrorBoundary>
              <TabErrorBoundary tabName="Achievements"><Achievements /></TabErrorBoundary>
            </div>
          </TabsContent>
        </div>
      </Tabs>

      <PhonemeReference />
    </PageContainer>
  )
}
