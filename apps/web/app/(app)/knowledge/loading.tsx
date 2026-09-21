import { PageContainer } from '@/components/PageContainer'
import { KnowledgePageSkeleton } from '@/components/ui/PageSkeletons'

export default function KnowledgeLoading() {
  return (
    <PageContainer title="Knowledge" loading loadingContent={<KnowledgePageSkeleton />}>
      <></>
    </PageContainer>
  )
}
