'use client'

/**
 * Consciousness API docs — composition root.
 *
 * The endpoint dataset lives in `docs-endpoints`, the try-it runner in
 * `useDocsWorkbench`. Only the page shell belongs here: the flex frame
 * around the category sidebar and the endpoint list.
 */
import { PageContainer } from '@/components/PageContainer'
import { useDocsWorkbench } from './useDocsWorkbench'
import { DocsSidebar } from './docs-sidebar'
import { DocsEndpointList } from './docs-endpoint-list'

export default function ConsciousnessDocsPage() {
  const wb = useDocsWorkbench()
  const { t } = wb

  return (
    <PageContainer title={t('consciousness_docs.page_title')}>
      <div className="flex gap-6 p-6 min-h-[calc(100vh-4rem)]">
        <DocsSidebar wb={wb} />
        <DocsEndpointList wb={wb} />
      </div>
    </PageContainer>
  )
}
