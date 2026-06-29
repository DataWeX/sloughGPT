'use client'

interface ImportResultModalProps {
  result: { ok: number; fail: number; names: string[] }
  onClose: () => void
}

export default function ImportResultModal({ result, onClose }: ImportResultModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-card rounded-lg border shadow-lg w-full max-w-md mx-4 p-4" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Import complete</h3>
          <button className="text-muted-foreground hover:text-foreground text-xs" onClick={onClose}>Close</button>
        </div>
        <div className="text-xs text-muted-foreground mb-2">
          {result.ok} imported, {result.fail} failed
        </div>
        {result.names.length > 0 && (
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {result.names.map((name, i) => (
              <div key={i} className="text-xs bg-muted/40 rounded px-2 py-1 truncate">{name}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
