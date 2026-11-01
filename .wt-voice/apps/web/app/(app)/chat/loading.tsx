import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function ChatLoading() {
  return (
    <PageContainer title="Chat" loading loadingContent={<PageSkeleton cards={1} header={false} />}>
      <></>
    </PageContainer>
  )
}
