'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function PDFAnalysisCard() {
  const addToast = useToastStore(s => s.addToast)
  const pdfFileInputRef = useRef<HTMLInputElement>(null)
  const [pdfPath, setPdfPath] = useState('')
  const [pdfQuestion, setPdfQuestion] = useState('Summarize this document.')
  const [pdfAnalyzing, setPdfAnalyzing] = useState(false)
  const [pdfResult, setPdfResult] = useState<string | null>(null)
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [pdfPerPage, setPdfPerPage] = useState(false)

  const handlePDFUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) setPdfFile(file)
  }

  const handleAnalyzePDF = async () => {
    if (!pdfPath.trim() && !pdfFile) return
    setPdfAnalyzing(true)
    setPdfResult(null)
    try {
      if (pdfFile) {
        const result = await visualController.analyzePDFUpload(pdfFile, pdfQuestion, pdfPerPage)
        setPdfResult(result.analysis || (result.pages || []).map(p => p.text).join('\n\n---\n\n'))
      } else {
        const result = await visualController.analyzePDF({
          pdf_path: pdfPath.trim(),
          question: pdfQuestion,
          per_page: pdfPerPage,
        })
        setPdfResult(result.analysis || (result.pages || []).map(p => p.text).join('\n\n---\n\n'))
      }
      addToast('PDF analysis complete', 'success')
    } catch (err: any) {
      addToast(`PDF analysis failed: ${err.message}`, 'error')
    } finally {
      setPdfAnalyzing(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">PDF Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Analyze PDF documents using the visual model. Provide a server path or upload a file.
        </p>

        <div className="flex gap-2">
          <Button size="sm" variant={pdfFile ? 'outline' : 'default'} onClick={() => { setPdfFile(null); setPdfPath('') }}>
            Server Path
          </Button>
          <Button size="sm" variant={pdfFile ? 'default' : 'outline'} onClick={() => pdfFileInputRef.current?.click()}>
            Upload File
          </Button>
          <input ref={pdfFileInputRef} type="file" accept=".pdf" className="hidden" onChange={handlePDFUpload} />
        </div>

        {pdfFile ? (
          <p className="text-xs text-muted-foreground truncate">Selected: {pdfFile.name}</p>
        ) : (
          <Input
            value={pdfPath}
            onChange={(e) => setPdfPath(e.target.value)}
            placeholder="Server path to PDF — e.g. /home/user/doc.pdf"
            className="text-sm"
          />
        )}

        <Input
          value={pdfQuestion}
          onChange={(e) => setPdfQuestion(e.target.value)}
          placeholder="What to ask about this document?"
          className="text-sm"
        />

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={pdfPerPage} onChange={(e) => setPdfPerPage(e.target.checked)} className="rounded" />
            <span className="text-xs text-muted-foreground">Analyze per page</span>
          </label>
          <Button size="sm" onClick={handleAnalyzePDF} disabled={pdfAnalyzing || (!pdfPath.trim() && !pdfFile)} className="ml-auto">
            {pdfAnalyzing ? 'Analyzing...' : 'Analyze'}
          </Button>
        </div>

        {pdfResult && (
          <div className="rounded-lg bg-muted/50 p-3 text-sm leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">
            {pdfResult}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
