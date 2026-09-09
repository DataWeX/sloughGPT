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
import { usePhonemeShortcuts } from '@/hooks/usePhonemeShortcuts'

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
  })

  return (
    <PageContainer
      title="Phoneme Encoder"
      subtitle="Multi-language phoneme encoding, pronunciation scoring, and TTS"
      headerRight={
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Shortcuts:</span>
          <Kbd>Ctrl+1-9</Kbd>
          <span>switch tabs</span>
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

        <TabsContent value="encode">
          <EncodingCard />
        </TabsContent>

        <TabsContent value="score">
          <ScoringCard />
        </TabsContent>

        <TabsContent value="compare">
          <ComparisonCard />
        </TabsContent>

        <TabsContent value="practice">
          <PracticeCard />
        </TabsContent>

        <TabsContent value="batch">
          <BatchCard />
        </TabsContent>

        <TabsContent value="detect">
          <DetectLanguageCard />
        </TabsContent>

        <TabsContent value="quiz">
          <QuizCard />
        </TabsContent>

        <TabsContent value="synthesize">
          <SynthesisCard />
        </TabsContent>

        <TabsContent value="history">
          <HistoryCard />
        </TabsContent>
      </Tabs>
    </PageContainer>
  )
}
