export default function Loading() {
  return (
    <div className="sl-page mx-auto max-w-4xl space-y-4 animate-pulse">
      <div className="h-8 w-48 rounded bg-muted" />
      <div className="h-32 rounded-lg bg-muted" />
      <div className="grid grid-cols-2 gap-4">
        <div className="h-24 rounded-lg bg-muted" />
        <div className="h-24 rounded-lg bg-muted" />
      </div>
    </div>
  )
}
