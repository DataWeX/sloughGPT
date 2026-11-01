import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function RateLimitLoading() {
  return (
    <PageContainer title="Rate Limit" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
