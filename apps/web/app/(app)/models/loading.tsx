import { PageContainer } from '@/components/PageContainer'
import { ModelsPageSkeleton } from '@/components/ui/PageSkeletons'

export default function ModelsLoading() {
  return (
    <PageContainer title="Models" loading loadingContent={<ModelsPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
