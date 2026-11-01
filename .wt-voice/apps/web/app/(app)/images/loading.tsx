import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function ImagesLoading() {
  return (
    <PageContainer title="Images" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
