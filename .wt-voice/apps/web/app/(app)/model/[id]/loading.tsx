import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function ModelDetailLoading() {
  return (
    <PageContainer title="Model" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
