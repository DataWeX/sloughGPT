import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Skeleton } from '@sloughgpt/strui'

export default function MonitoringLoading() {
  return (
    <PageContainer title="System Health" loading loadingContent={
      <div className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-5 w-12" />
            </CardContent></Card>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card><CardContent className="p-4 space-y-3">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-40 w-full rounded" />
          </CardContent></Card>
          <Card><CardContent className="p-4 space-y-3">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-40 w-full rounded" />
          </CardContent></Card>
        </div>
      </div>
    }>
      <></>
    </PageContainer>
  )
}
