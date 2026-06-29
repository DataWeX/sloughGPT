'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { IconUpload } from '@/components/ui'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function ImageInferenceCard() {
  const addToast = useToastStore(s => s.addToast)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [inferImage, setInferImage] = useState<string | null>(null)
  const [inferPrompt, setInferPrompt] = useState('Describe this image in detail.')
  const [inferResult, setInferResult] = useState<string | null>(null)
  const [inferRunning, setInferRunning] = useState(false)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setInferImage(reader.result as string)
    reader.readAsDataURL(file)
  }

  const handleInfer = async () => {
    if (!inferImage) return
    setInferRunning(true)
    setInferResult(null)
    try {
      const result = await visualController.visualInference({ image_path: inferImage, max_len: 256 })
      setInferResult(result.text)
      addToast(`Visual inference: ${result.elapsed_ms}ms`, 'success')
    } catch (err: any) {
      addToast(`Inference failed: ${err.message}`, 'error')
    } finally {
      setInferRunning(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Image Inference</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-3">
          <div className="flex-1">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageUpload}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              className="w-full"
            >
              <IconUpload className="h-4 w-4 mr-2" />
              {inferImage ? 'Change Image' : 'Upload Image'}
            </Button>
            {inferImage && (
              <img src={inferImage} alt="Upload" className="mt-2 h-32 w-32 object-cover rounded-lg border" />
            )}
          </div>
          <div className="flex-1 space-y-2">
            <Input
              value={inferPrompt}
              onChange={(e) => setInferPrompt(e.target.value)}
              placeholder="What to ask about this image?"
              className="text-sm"
            />
            <Button
              size="sm"
              onClick={handleInfer}
              disabled={!inferImage || inferRunning}
              className="w-full"
            >
              {inferRunning ? 'Analyzing...' : 'Run Inference'}
            </Button>
          </div>
        </div>
        {inferResult && (
          <div className="rounded-lg bg-muted/50 p-3 text-sm font-mono leading-relaxed">
            {inferResult}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
