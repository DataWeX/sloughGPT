'use client'
export const dynamic = 'force-dynamic'

/**
 * Knowledge — composition root.
 *
 * State and IO live in `useKnowledgeWorkbench`; rendering lives in the
 * `knowledge-*` siblings next to this file. Only the page shell belongs
 * here: header chrome, the KPI strip and section wiring.
 */
import { useSearchParams } from '@/vite/next-compat/navigation'
import { PageContainer } from '@/components/PageContainer'
import { FoldSection } from '@sloughgpt/strui'
import { MemoryCard } from '@/components/knowledge/MemoryCard'
import { ReadMyFilesCard } from '@/components/knowledge/ReadMyFilesCard'
import { KnowledgeIntelligenceCard } from '@/components/knowledge/KnowledgeIntelligenceCard'
import { MemorySettingsCard } from '@/components/knowledge/MemorySettingsCard'
import { LearnSection } from '@/components/learn/LearnSection'
import { useKnowledgeWorkbench } from './useKnowledgeWorkbench'
import { KnowledgeHeaderActions, KnowledgeToolbar } from './knowledge-chrome'
import { KnowledgeStatsStrip } from './knowledge-stats-strip'
import { KnowledgeAdvancedSection } from './knowledge-advanced-section'
import { KnowledgeItemsList } from './knowledge-items-list'
import { KnowledgeDetailFold } from './knowledge-detail-fold'
import { KnowledgeDeleteDialog } from './knowledge-delete-dialog'
import { KnowledgeBatchDeleteDialog } from './knowledge-batch-delete-dialog'
import { KnowledgeBulkTopicDialog } from './knowledge-bulk-topic-dialog'
import { KnowledgeAddDialog } from './knowledge-add-dialog'

export default function KnowledgePage() {
  const wb = useKnowledgeWorkbench()
  const { addToast, fetchData } = wb
  // ?id=-driven detail fold — the old /knowledge/[id] route composed in here
  // (route-map card 7/7; card-3 /models pattern). List click sets ?id=,
  // Back/Delete clear it by navigating to bare /knowledge.
  const searchParams = useSearchParams()
  const selectedId = searchParams.get('id')

  return (
    <PageContainer
      title="Knowledge"
      subtitle="Manage facts the AI remembers"
      headerRight={<KnowledgeHeaderActions wb={wb} />}
      toolbar={<KnowledgeToolbar wb={wb} />}
    >
      <ReadMyFilesCard addToast={addToast} onIngested={fetchData} />

      <KnowledgeStatsStrip wb={wb} />

      <KnowledgeAdvancedSection wb={wb} />

      {selectedId && (
        <FoldSection heading="Item detail" open>
          <KnowledgeDetailFold id={selectedId} />
        </FoldSection>
      )}

      <KnowledgeItemsList wb={wb} />

      <FoldSection heading="More — settings & intelligence">
        <div className="space-y-4">
          <MemoryCard />
          <MemorySettingsCard />
          <LearnSection />
          <KnowledgeIntelligenceCard />
        </div>
      </FoldSection>

      {/* ═══════════════ DIALOGS ═══════════════ */}
      <KnowledgeDeleteDialog wb={wb} />
      <KnowledgeBatchDeleteDialog wb={wb} />
      <KnowledgeBulkTopicDialog wb={wb} />
      <KnowledgeAddDialog wb={wb} />
    </PageContainer>
  )
}
