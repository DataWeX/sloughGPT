export default function ChatLoading() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] space-y-4">
      <span className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" role="status" aria-label="Loading" />
      <p className="text-sm text-muted-foreground">Loading chat...</p>
    </div>
  )
}
