import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function BenchmarkLoading() {
  return (
    <PageContainer title="Benchmark" subtitle="Model evaluation metrics" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
