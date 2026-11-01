'use client'

import { useCallback, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button, Card, CardContent, IconUpload, Spinner } from '@sloughgpt/strui'
import { kbController } from '@/lib/kb-controller'

const SUGGESTED_QUESTIONS = [
  'Summarize this',
  'What are the key points?',
  'Explain this in simple terms',
] as const

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
  onIngested?: () => void
}

export function ReadMyFilesCard({ addToast, onIngested }: Props) {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [reading, setReading] = useState(false)
  const [result, setResult] = useState<{ filename: string; totalChunks: number } | null>(null)

  const ingest = useCallback(
    async (file: File) => {
      const lower = file.name.toLowerCase()
      if (lower.endsWith('.docx') || lower.endsWith('.doc')) {
        addToast("Word files aren't supported yet — upload a PDF or a text file", 'error')
        return
      }
      setReading(true)
      setResult(null)
      try {
        const res = await kbController.ingestFile(file)
        setResult({ filename: res.filename, totalChunks: res.total_chunks })
        addToast(`Read ${file.name}`, 'success')
        onIngested?.()
      } catch (err) {
        addToast(err instanceof Error ? err.message : "Couldn't read that file", 'error')
      } finally {
        setReading(false)
      }
    },
    [addToast, onIngested],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files?.[0]
      if (file) void ingest(file)
    },
    [ingest],
  )

  const askQuestion = useCallback(
    (question: string) => {
      router.push(`/chat?q=${encodeURIComponent(question)}`)
    },
    [router],
  )

  return (
    <Card
      className={dragOver ? 'border-primary/60 border-dashed' : 'border-dashed'}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <CardContent className="p-5 space-y-3">
        <div className="flex items-center gap-2">
          <IconUpload className="h-4 w-4 text-primary" aria-hidden />
          <p className="text-base font-medium">Read my files</p>
          <p className="text-xs text-muted-foreground">
            PDF or text files — drop one here or click to upload
          </p>
        </div>

        <div
          role="button"
          tabIndex={0}
          aria-label="Upload a file to read"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          className="flex flex-col items-center justify-center gap-2 rounded-lg border border-border/40 bg-card/50 px-4 py-6 text-center transition-all duration-200 cursor-pointer hover:border-primary/50 hover:bg-primary/[0.03] focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none active:scale-[0.99]"
        >
          {reading ? (
            <>
              <Spinner className="h-4 w-4" />
              <p className="text-sm text-muted-foreground">Reading your file...</p>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              Drop a file here or click to upload
            </p>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,.csv,.json,.docx,.doc"
          className="hidden"
          aria-label="Choose a file to read"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void ingest(file)
            e.target.value = ''
          }}
        />

        {result && (
          <div className="space-y-2 animate-in fade-in" aria-live="polite">
            <p className="text-sm">
              Got it — I read {result.totalChunks} piece{result.totalChunks === 1 ? '' : 's'} from{' '}
              <span className="font-medium">{result.filename}</span>. What do you want to know?
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <Button
                  key={q}
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs transition-all duration-200 active:scale-[0.98]"
                  onClick={() => askQuestion(q)}
                >
                  {q}
                </Button>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
