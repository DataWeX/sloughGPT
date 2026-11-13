import { PageContainer } from '@/components/PageContainer'
import { VectorPageSkeleton } from '@/components/ui/PageSkeletons'

export default function VectorLoading() {
  return (
    <PageContainer title="Vector" loading loadingContent={<VectorPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
