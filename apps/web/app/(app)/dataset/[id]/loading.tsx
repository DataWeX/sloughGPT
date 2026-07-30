export default function DatasetDetailLoading() {
  return (
    <div className="sl-page mx-auto max-w-4xl flex flex-col items-center justify-center min-h-[60vh] space-y-6">
      <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" role="status" aria-label="Loading" />
      <p className="text-sm text-muted-foreground">Loading dataset...</p>
    </div>
  )
}
