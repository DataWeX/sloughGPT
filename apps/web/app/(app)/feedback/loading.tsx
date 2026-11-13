import { PageContainer } from '@/components/PageContainer'
import { FeedbackPageSkeleton } from '@/components/ui/PageSkeletons'

export default function FeedbackLoading() {
  return (
    <PageContainer title="Feedback" subtitle="Analytics & management" loading loadingContent={<FeedbackPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
