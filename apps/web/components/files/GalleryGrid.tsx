import { PUBLIC_API_URL } from '@/lib/config'
import type { GalleryImage } from '@/lib/images-controller'

/**
 * Generated-image gallery grid; src = API origin + path so it resolves in
 * dev (vite has no /images proxy) and behind the gateway in production —
 * same pattern the /images page established.
 *
 * Presentational only — the owning page fetches (imagesController.gallery).
 * Shared by the /files Images tab and the live /developer Images tab.
 */
export function GalleryGrid({ images, loading }: { images: GalleryImage[]; loading: boolean }) {
  if (loading) return <div className="h-32 animate-pulse bg-muted/20 rounded-lg" />
  if (images.length === 0) return <p className="text-[10px] text-muted-foreground/60">No generated images yet.</p>
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2" data-testid="gallery-grid">
      {images.map(img => (
        <figure key={img.id} className="rounded-lg border border-border/40 overflow-hidden">
          <img
            src={`${PUBLIC_API_URL}${img.path}`}
            alt={img.id}
            loading="lazy"
            className="w-full aspect-square object-cover"
          />
          <figcaption className="px-1.5 py-1 text-[9px] text-muted-foreground/60 font-mono truncate">
            {img.id} · {new Date(img.created * 1000).toLocaleDateString()}
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
