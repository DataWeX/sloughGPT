import { PageContainer } from '@/components/PageContainer'
import { ComparePageSkeleton } from '@/components/ui/PageSkeletons'

export default function CompareLoading() {
  return (
    <PageContainer title="Compare" loading loadingContent={<ComparePageSkeleton />}>
      <></>
    </PageContainer>
  )
}
