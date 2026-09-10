'use client'
export const dynamic = 'force-dynamic'

import { useState, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Tabs, TabsList, TabsTrigger, TabsContent, Kbd } from '@sloughgpt/strui'
import EncodingCard from '@/components/phoneme/EncodingCard'
import ScoringCard from '@/components/phoneme/ScoringCard'
import ComparisonCard from '@/components/phoneme/ComparisonCard'
import PracticeCard from '@/components/phoneme/PracticeCard'
import BatchCard from '@/components/phoneme/BatchCard'
import DetectLanguageCard from '@/components/phoneme/DetectLanguageCard'
import QuizCard from '@/components/phoneme/QuizCard'
import SynthesisCard from '@/components/phoneme/SynthesisCard'
import HistoryCard from '@/components/phoneme/HistoryCard'
import PhonemeReference from '@/components/phoneme/PhonemeReference'
import TabErrorBoundary from '@/components/phoneme/TabErrorBoundary'
import { usePhonemeShortcuts } from '@/hooks/usePhonemeShortcuts'
import { usePhonemeStore } from '@/lib/phoneme-store'

const TABS = [
  { value: 'encode', label: 'Encode', shortcut: '1' },
  { value: 'score', label: 'Score', shortcut: '2' },
  { value: 'compare', label: 'Compare', shortcut: '3' },
  { value: 'practice', label: 'Practice', shortcut: '4' },
  { value: 'batch', label: 'Batch', shortcut: '5' },
  { value: 'detect', label: 'Detect', shortcut: '6' },
  { value: 'quiz', label: 'Quiz', shortcut: '7' },
  { value: 'synthesize', label: 'Synthesize', shortcut: '8' },
  { value: 'history', label: 'History', shortcut: '9' },
] as const

export default function PhonemePage() {
  const [activeTab, setActiveTab] = useState('encode')
  const triggerRandomWord = usePhonemeStore(s => s.triggerRandomWord)
  const clearHistory = usePhonemeStore(s => s.clearHistory)

  const goToTab = useCallback((value: string) => {
    setActiveTab(value)
  }, [])

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
  })

  return (
    <PageContainer
      title="Phoneme Encoder"
      subtitle="Multi-language phoneme encoding, pronunciation scoring, and TTS"
      headerRight={
        <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
          <Kbd>Ctrl+1-9</Kbd>
          <span>tabs</span>
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
            <TabErrorBoundary tabName="Practice"><PracticeCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="batch">
            <TabErrorBoundary tabName="Batch"><BatchCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="detect">
            <TabErrorBoundary tabName="Detect"><DetectLanguageCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="quiz">
            <TabErrorBoundary tabName="Quiz"><QuizCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="synthesize">
            <TabErrorBoundary tabName="Synthesize"><SynthesisCard /></TabErrorBoundary>
          </TabsContent>

          <TabsContent value="history">
            <TabErrorBoundary tabName="History"><HistoryCard /></TabErrorBoundary>
          </TabsContent>
        </div>
      </Tabs>

      <PhonemeReference />
    </PageContainer>
  )
}
