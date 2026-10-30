'use client'

import { memo, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { ImageLightbox } from './ImageLightbox'
import type { ImageAttachment } from './../input/ImageUpload'

interface MessageImagesProps {
  images: ImageAttachment[]
  role: 'user' | 'assistant'
}

export const MessageImages = memo(function MessageImages({ images, role }: MessageImagesProps) {
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)
  const [failed, setFailed] = useState<Record<string, boolean>>({})

  if (!images || images.length === 0) return null

  return (
    <>
      <div className={cn('flex gap-2 mb-3 flex-wrap', role === 'user' && 'flex-row-reverse')}>
        {images.map((img) =>
          failed[img.id] ? (
            // Thumbnail failed to decode (bad dataUrl, evicted blob, etc.) —
            // show a named fallback instead of a bare broken-image glyph.
            <div
              key={img.id}
              className="max-h-24 min-w-[7rem] px-3 py-2 rounded-xl border border-dashed border-current/30 bg-muted/40 text-[10px] font-medium flex flex-col items-start justify-center gap-0.5"
              role="img"
              aria-label={`Image unavailable: ${img.name}`}
            >
              <span className="text-muted-foreground">Image unavailable</span>
              <span className="text-foreground/70 truncate max-w-[10rem]">{img.name}</span>
            </div>
          ) : (
            <button
              key={img.id}
              type="button"
              onClick={() => setLightboxSrc(img.dataUrl)}
              className="active:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset p-0 border-0 bg-transparent rounded-xl cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/40 hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              aria-label={`View ${img.name} full size`}
            >
              <img
                src={img.dataUrl}
                alt={img.name}
                onError={() => setFailed((prev) => ({ ...prev, [img.id]: true }))}
                className="max-h-24 w-auto rounded-xl object-cover border border-current/20 shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200"
              />
            </button>
          ),
        )}
      </div>
      {lightboxSrc && (
        <ImageLightbox src={lightboxSrc} alt="Image preview" onClose={() => setLightboxSrc(null)} />
      )}
    </>
  )
})
