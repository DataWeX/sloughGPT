import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function KbLoading() {
  return (
    <PageContainer title="Knowledge Base" loading loadingContent={<PageSkeleton cards={2} header={false} />}>
      <></>
    </PageContainer>
  )
}
