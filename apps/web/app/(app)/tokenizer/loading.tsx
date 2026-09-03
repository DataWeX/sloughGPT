import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Skeleton } from '@sloughgpt/strui'

export default function TokenizerLoading() {
  return (
    <PageContainer title="Tokenizer" subtitle="BPE tokenizer" loading loadingContent={
      <div className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-5 w-14" />
            </CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="p-4 space-y-3">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-9 w-full rounded-md" />
          <Skeleton className="h-24 w-full rounded" />
        </CardContent></Card>
      </div>
    }>
      <></>
    </PageContainer>
  )
}
