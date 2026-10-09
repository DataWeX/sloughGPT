import { PageContainer } from '@/components/PageContainer'
import { DatasetsPageSkeleton } from '@/components/ui/PageSkeletons'

export default function DatasetsLoading() {
  return (
    <PageContainer title="Datasets" loading loadingContent={<DatasetsPageSkeleton />}>
      <></>
    </PageContainer>
  )
}
