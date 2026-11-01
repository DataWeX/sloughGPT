import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function AutoTrainLoading() {
  return (
    <PageContainer title="Auto-Train" loading loadingContent={<PageSkeleton cards={4} header={false} />}>
      <></>
    </PageContainer>
  )
}
