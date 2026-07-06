'use client'

import { useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'
import { useToastStore } from '@/lib/toast-store'

interface ExportDropdownProps {
  jobId: string
  checkpoint: string
}

export function ExportDropdown({ jobId, checkpoint }: ExportDropdownProps) {
  const addToast = useToastStore(s => s.addToast)
  const [format, setFormat] = useState('pt')
  const [exporting, setExporting] = useState(false)

  const handleExport = async () => {
    setExporting(true)
    try {
      const blob = await trainingJobsController.downloadTrainingJob(jobId)
      const filename = checkpoint.split('/').pop() || `model.${format}`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
      URL.revokeObjectURL(url)
      addToast(`Downloaded ${filename}`, 'success')
    } catch (err) {
      addToast(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error')
    } finally {
      setExporting(false)
    }
  }

  const formats = [
    { id: 'pt', name: 'PyTorch (.pt)', desc: 'Standard PyTorch format' },
    { id: 'sou', name: 'Soul (.soul)', desc: 'Man with personality' },
    { id: 'safetensors', name: 'SafeTensors', desc: 'Safe, memory-mapped' },
    { id: 'onnx', name: 'ONNX (.onnx)', desc: 'Cross-platform inference' },
    { id: 'gguf', name: 'GGUF', desc: 'Mobile/embedded (llama.cpp)' },
  ]

  return (
    <div className="flex gap-2 items-center">
      <Select value={format} onValueChange={setFormat}>
        <SelectTrigger className="text-xs h-8">
          <SelectValue placeholder="Select format..." />
        </SelectTrigger>
        <SelectContent>
          {formats.map((f) => (
            <SelectItem key={f.id} value={f.id}>{f.name}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        size="sm"
        variant="outline"
        onClick={handleExport}
        disabled={exporting}
      >
        {exporting ? 'Exporting...' : 'Export'}
      </Button>
    </div>
  )
}
