import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function VmLoading() {
  return (
    <PageContainer title="VM" loading loadingContent={<PageSkeleton cards={6} header={false} />}>
      <></>
    </PageContainer>
  )
}
