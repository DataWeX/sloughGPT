import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function ExportLoading() {
  return (
    <PageContainer title="Export" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
