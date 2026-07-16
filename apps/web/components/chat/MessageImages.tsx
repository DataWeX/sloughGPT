'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { ImageLightbox } from './ImageLightbox'
import type { ImageAttachment } from './ImageUpload'

interface MessageImagesProps {
  images: ImageAttachment[]
  role: 'user' | 'assistant'
}

export function MessageImages({ images, role }: MessageImagesProps) {
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)

  if (!images || images.length === 0) return null

  return (
    <>
      <div className={cn(
        "flex gap-2 mb-3 flex-wrap",
        role === 'user' && "flex-row-reverse"
      )}>
        {images.map((img) => (
          <button
            key={img.id}
            type="button"
            onClick={() => setLightboxSrc(img.dataUrl)}
            className="p-0 border-0 bg-transparent rounded-xl cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/40"
            aria-label={`View ${img.name} full size`}
          >
            <img
              src={img.dataUrl}
              alt={img.name}
              className="h-24 w-24 rounded-xl object-cover border border-current/20 shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200"
            />
          </button>
        ))}
      </div>
      {lightboxSrc && (
        <ImageLightbox
          src={lightboxSrc}
          alt="Image preview"
          onClose={() => setLightboxSrc(null)}
        />
      )}
    </>
  )
}
