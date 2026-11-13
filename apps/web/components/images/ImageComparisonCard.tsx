'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface CompareImage {
  id: string
  label: string
  src: string
}

interface ImageComparisonCardProps {
  images?: CompareImage[]
}

export function ImageComparisonCard({ images = [] }: ImageComparisonCardProps) {
  const [selected, setSelected] = useState<[string | null, string | null]>([null, null])
  const [sliderPos, setSliderPos] = useState(50)
  const [mode, setMode] = useState<'side-by-side' | 'slider' | 'overlay'>('side-by-side')

  const imgA = images.find(i => i.id === selected[0])
  const imgB = images.find(i => i.id === selected[1])

  return (
    <Card data-testid="image-comparison">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Image Comparison</CardTitle>
          <div className="flex gap-1">
            {(['side-by-side', 'slider', 'overlay'] as const).map(m => (
              <Button
                key={m}
                size="sm"
                variant={mode === m ? 'default' : 'ghost'}
                className="text-[10px]"
                onClick={() => setMode(m)}
              >
                {m === 'side-by-side' ? 'SxS' : m === 'slider' ? 'Slider' : 'Overlay'}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-3">
          <select
            className="flex-1 text-xs border border-border rounded px-2 py-1 bg-background"
            value={selected[0] ?? ''}
            onChange={e => setSelected([e.target.value || null, selected[1]])}
          >
            <option value="">Select image A</option>
            {images.map(i => <option key={i.id} value={i.id}>{i.label}</option>)}
          </select>
          <select
            className="flex-1 text-xs border border-border rounded px-2 py-1 bg-background"
            value={selected[1] ?? ''}
            onChange={e => setSelected([selected[0], e.target.value || null])}
          >
            <option value="">Select image B</option>
            {images.map(i => <option key={i.id} value={i.id}>{i.label}</option>)}
          </select>
        </div>

        {!imgA && !imgB && (
          <div className="text-center py-8 text-sm text-muted-foreground">
            Select two images to compare
          </div>
        )}

        {imgA && imgB && mode === 'side-by-side' && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] text-muted-foreground mb-1">{imgA.label}</div>
              <img src={imgA.src} alt={imgA.label} className="w-full rounded border border-border" />
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground mb-1">{imgB.label}</div>
              <img src={imgB.src} alt={imgB.label} className="w-full rounded border border-border" />
            </div>
          </div>
        )}

        {imgA && imgB && mode === 'slider' && (
          <div className="relative overflow-hidden rounded border border-border">
            <img src={imgA.src} alt={imgA.label} className="w-full" />
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ width: `${sliderPos}%` }}
            >
              <img
                src={imgB.src}
                alt={imgB.label}
                className="w-full"
                style={{ width: `${100 / (sliderPos / 100)}%`, maxWidth: 'none' }}
              />
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={sliderPos}
              onChange={e => setSliderPos(Number(e.target.value))}
              className="absolute bottom-2 left-0 right-0 w-full accent-primary"
            />
          </div>
        )}

        {imgA && imgB && mode === 'overlay' && (
          <div className="relative overflow-hidden rounded border border-border">
            <img src={imgA.src} alt={imgA.label} className="w-full" />
            <img
              src={imgB.src}
              alt={imgB.label}
              className="absolute inset-0 w-full h-full object-cover opacity-50 mix-blend-difference"
            />
          </div>
        )}

        {imgA && !imgB && <img src={imgA.src} alt={imgA.label} className="w-full rounded border border-border" />}
        {!imgA && imgB && <img src={imgB.src} alt={imgB.label} className="w-full rounded border border-border" />}
      </CardContent>
    </Card>
  )
}
