import { PageContainer } from '@/components/PageContainer'
import { MemoryPageSkeleton } from '@/components/ui/PageSkeletons'

export default function MemoryLoading() {
  return (
    <PageContainer title="Memory" loading loadingContent={<MemoryPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
