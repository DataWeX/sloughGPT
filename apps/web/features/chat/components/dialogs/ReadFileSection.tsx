'use client'

interface ReadFileSectionProps {
  readLoading: boolean
  readFileData: { text: string; filename: string; pages: number } | null
  onFileSelected: (file: File) => void
  onRemove: () => void
}

export default function ReadFileSection({ readLoading, readFileData, onFileSelected, onRemove }: ReadFileSectionProps) {
  if (!readFileData) {
    return (
      <div className="px-3 py-2 border-b border-border/10 bg-muted/5">
        <div
          className="flex flex-col items-center gap-2 py-6 border-2 border-dashed border-border/30 rounded-lg text-center cursor-pointer hover:border-primary/40 hover:bg-muted/10 transition-colors"
          onDragOver={e => e.preventDefault()}
          onDrop={async e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) onFileSelected(f) }}
        >
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv,.json"
            className="hidden"
            id="read-file-input"
            onChange={e => { const f = e.target.files?.[0]; if (f) onFileSelected(f) }}
          />
          <label htmlFor="read-file-input" className="cursor-pointer flex flex-col items-center gap-1">
            <span className="text-2xl">📄</span>
            <span className="text-sm font-medium">{readLoading ? 'Reading your file...' : 'Drop a file here or click to upload'}</span>
            <span className="text-[11px] text-muted-foreground">PDF, Word, TXT, MD, CSV, JSON</span>
          </label>
        </div>
      </div>
    )
  }

  return (
    <div className="px-3 py-2 border-b border-border/10 bg-muted/5">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-base">📄</span>
        <span className="font-medium truncate max-w-[200px]">{readFileData.filename}</span>
        {readFileData.pages > 1 && <span className="text-xs text-muted-foreground">({readFileData.pages} pages)</span>}
        <button
          type="button"
          onClick={onRemove}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-muted/10"
        >
          Remove
        </button>
      </div>
    </div>
  )
}
