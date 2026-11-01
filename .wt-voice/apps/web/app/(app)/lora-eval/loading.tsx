import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function LoraEvalLoading() {
  return (
    <PageContainer title="LoRA Eval" loading loadingContent={<PageSkeleton cards={4} header={false} />}>
      <></>
    </PageContainer>
  )
}
