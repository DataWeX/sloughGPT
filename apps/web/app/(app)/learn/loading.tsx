export default function LearnLoading() {
  return (
    <div className="sl-page mx-auto max-w-4xl">
      <div className="mb-4">
        <div className="h-8 w-32 animate-pulse rounded bg-muted" />
      </div>
      <div className="space-y-3">
        {[1, 2, 3].map(i => (
          <div key={i} className="sl-card p-4">
            <div className="h-4 w-48 animate-pulse rounded bg-muted mb-2" />
            <div className="h-3 w-full animate-pulse rounded bg-muted" />
          </div>
        ))}
      </div>
    </div>
  )
}
