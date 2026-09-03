import { Skeleton } from '@sloughgpt/strui'

export default function KanbanLoading() {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-8 w-32" />
        <div className="flex gap-2">
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-8 w-8" />
        </div>
      </div>
      <div className="flex gap-3 overflow-x-auto pb-4">
        {Array.from({ length: 4 }).map((_, col) => (
          <div key={col} className="flex-shrink-0 w-64 space-y-2">
            <Skeleton className="h-6 w-20" />
            {Array.from({ length: 3 }).map((_, card) => (
              <div key={card} className="rounded-lg border border-border/30 bg-card p-3 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-1/2" />
                <div className="flex gap-1 mt-2">
                  <Skeleton className="h-4 w-12" />
                  <Skeleton className="h-4 w-16" />
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
