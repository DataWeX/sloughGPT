import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Skeleton } from '@sloughgpt/strui'

export default function SoulsLoading() {
  return (
    <PageContainer title="Souls" subtitle="Personality management" loading loadingContent={
      <div className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-5 w-12" />
            </CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="p-4 space-y-3">
          <Skeleton className="h-4 w-28" />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="p-2.5 rounded-lg border border-border/40 space-y-1.5">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-3/4" />
              </div>
            ))}
          </div>
        </CardContent></Card>
      </div>
    }>
      <></>
    </PageContainer>
  )
}
