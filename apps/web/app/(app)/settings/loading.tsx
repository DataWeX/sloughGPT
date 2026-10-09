import { PageContainer } from '@/components/PageContainer'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

export default function SettingsLoading() {
  return (
    <PageContainer title="Settings" loading loadingContent={<PageSkeleton cards={4} header={false} />}>
      <></>
    </PageContainer>
  )
}
