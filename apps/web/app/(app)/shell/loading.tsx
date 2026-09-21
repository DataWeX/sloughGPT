import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function ShellLoading() {
  return (
    <PageContainer title="Shell" loading loadingContent={<PageSkeleton cards={1} header={false} />}>
      <></>
    </PageContainer>
  )
}
