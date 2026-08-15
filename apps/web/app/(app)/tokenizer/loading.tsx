import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function TokenizerLoading() {
  return (
    <PageContainer title="Tokenizer" subtitle="BPE tokenizer" loading loadingContent={<PageSkeleton cards={3} header={false} />}>
      <></>
    </PageContainer>
  )
}
