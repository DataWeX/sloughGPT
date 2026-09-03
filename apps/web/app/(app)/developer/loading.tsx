import { Skeleton } from '@sloughgpt/strui'

export default function DeveloperLoading() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-32" />
      <Skeleton className="h-4 w-48" />
      <div className="flex gap-1 border-b border-border/30">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-20" />
        ))}
      </div>
      <div className="rounded-lg border border-border/30 bg-card p-6 space-y-3">
        <Skeleton className="h-5 w-24" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-3/4" />
        <Skeleton className="h-8 w-32 mt-4" />
      </div>
    </div>
  )
}
