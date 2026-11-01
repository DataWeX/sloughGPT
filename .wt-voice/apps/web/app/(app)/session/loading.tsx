import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function SessionLoading() {
  return (
    <PageContainer title="Session" loading loadingContent={<PageSkeleton cards={2} header={false} />}>
      <></>
    </PageContainer>
  )
}
