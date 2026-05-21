'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="sl-page mx-auto max-w-4xl flex flex-col items-center justify-center gap-4 py-16">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-destructive/10">
        <span className="text-lg font-bold text-destructive">!</span>
      </div>
      <h1 className="text-base font-semibold">Something went wrong</h1>
      <p className="text-sm text-muted-foreground text-center max-w-sm">
        {error.message || 'An unexpected error occurred.'}
      </p>
      <button
        onClick={reset}
        className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground px-4 h-9 text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Try again
      </button>
    </div>
  )
}
