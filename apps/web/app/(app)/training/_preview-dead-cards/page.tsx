'use client'
export const dynamic = 'force-dynamic'

import { PageContainer } from '@/components/PageContainer'
import { SectionHeader } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import type { Checkpoint } from '@/lib/souls-controller'
import { WebhooksCard } from '@/components/training/WebhooksCard'
import { TrainingDataCard } from '@/components/training/TrainingDataCard'
import { TrainingSummaryCard } from '@/components/training/TrainingSummaryCard'
import { TrainingTipsCard } from '@/components/training/TrainingTipsCard'
import { TrainingLogCard } from '@/components/training/TrainingLogCard'

const MOCK_CHECKPOINTS: Checkpoint[] = [
  { name: 'soul-0012', soul: 'soul-0012', loss: 1.8432, steps: 1200, epochs: 3, size_mb: 42, training_duration_s: 640, vocab_size: 32000, avg_quality: 4.2, model_type: 'slonet-s', training_dataset: 'conversations-v2', is_loaded: true },
  { name: 'soul-0011', soul: 'soul-0011', loss: 2.1004, steps: 800, epochs: 2, size_mb: 42, training_duration_s: 430, vocab_size: 32000, avg_quality: 3.6, model_type: 'slonet-s', training_dataset: 'conversations-v2' },
  { name: 'soul-0010', soul: 'soul-0010', loss: 2.4517, steps: 600, epochs: 2, size_mb: 41, training_duration_s: 380, vocab_size: 28000, avg_quality: 3.1, model_type: 'slonet-xs' },
  { name: 'soul-0009', soul: 'soul-0009', loss: 2.9188, steps: 400, epochs: 1, size_mb: 41, training_duration_s: 250, vocab_size: 28000, avg_quality: 2.8, model_type: 'slonet-xs' },
  { name: 'soul-0008', soul: 'soul-0008', steps: 300, epochs: 1, size_mb: 18, training_duration_s: 190, model_type: 'slonet-nano' },
]

export default function DeadCardsPreviewPage() {
  const addToast = useToastStore((s) => s.addToast)

  return (
    <PageContainer
      title="TEMP preview — dead training cards"
      subtitle="Render-only scaffold for card review; delete after screenshots."
      className="items-start"
    >
      <SectionHeader
        title="1. Training summary"
        description="components/training/TrainingSummaryCard.tsx — 105 lines (+143 test) — zero consumers"
      />
      <TrainingSummaryCard checkpoints={MOCK_CHECKPOINTS} loading={false} />

      <SectionHeader
        title="2. Training tips"
        description="components/training/TrainingTipsCard.tsx — 128 lines (+69 test) — zero consumers"
      />
      <TrainingTipsCard checkpoints={MOCK_CHECKPOINTS} loading={false} />

      <SectionHeader
        title="3. Training log"
        description="components/training/TrainingLogCard.tsx — 220 lines (+228 test) — zero consumers"
      />
      <TrainingLogCard trainingRunning={false} />

      <SectionHeader
        title="4. Webhooks"
        description="components/training/WebhooksCard.tsx — 192 lines (+256 test) — zero consumers; UX_FLOWS bans webhooks"
      />
      <WebhooksCard addToast={addToast} />

      <SectionHeader
        title="5. Training data"
        description="components/training/TrainingDataCard.tsx — 192 lines (+94 test) — zero consumers"
      />
      <TrainingDataCard addToast={addToast} />
    </PageContainer>
  )
}
