import { PageContainer } from '@/components/PageContainer'
import { AgentsPageSkeleton } from '@/components/ui/PageSkeletons'

export default function AgentsLoading() {
  return (
    <PageContainer title="Agents" loading loadingContent={<AgentsPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
