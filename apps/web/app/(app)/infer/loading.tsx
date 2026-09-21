import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function InferLoading() {
  return (
    <PageContainer title="Infer" loading loadingContent={<PageSkeleton cards={2} header={false} />}>
      <></>
    </PageContainer>
  )
}
