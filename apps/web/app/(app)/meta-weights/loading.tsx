import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function MetaWeightsLoading() {
  return (
    <PageContainer title="Meta Weights" loading loadingContent={<PageSkeleton cards={2} header={false} />}>
      <></>
    </PageContainer>
  )
}
