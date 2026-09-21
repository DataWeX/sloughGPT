import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function EvaluateLoading() {
  return (
    <PageContainer title="Evaluate" loading loadingContent={<PageSkeleton cards={2} header={false} />}>
      <></>
    </PageContainer>
  )
}
