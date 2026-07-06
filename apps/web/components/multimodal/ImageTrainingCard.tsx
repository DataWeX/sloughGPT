'use client'

import { useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconUpload } from '@sloughgpt/strui'

interface ImageTrainingCardProps {
  uploading: boolean
  onUpload: (file: File) => void
}

export default function ImageTrainingCard({ uploading, onUpload }: ImageTrainingCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Image Training</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">Upload an image to train the vision model. It learns features and generates a caption automatically.</p>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="h-8 text-xs" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            <IconUpload className="h-3.5 w-3.5 mr-1" />
            {uploading ? 'Training…' : 'Upload image'}
          </Button>
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onUpload(f) }} />
        </div>
      </CardContent>
    </Card>
  )
}
