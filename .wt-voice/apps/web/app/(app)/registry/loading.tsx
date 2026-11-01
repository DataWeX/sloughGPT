import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function RegistryLoading() {
  return (
    <PageContainer title="Registry" subtitle="Model registry" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
