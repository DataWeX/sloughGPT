import { apiPost, apiGet } from '@/lib/http-client'

export interface GenerateImageResponse {
  image: string
  style: string
  prompt: string
  id: string
}

export interface GalleryImage {
  id: string
  path: string
  created: number
}

export interface GalleryResponse {
  images: GalleryImage[]
}

export type ImageStyle = 'realistic' | 'cartoon' | 'watercolor' | 'sketch' | 'fantasy'

export const imagesController = {
  /** Generate an image from a text prompt using the given style. */
  async generate(prompt: string, style: ImageStyle = 'realistic'): Promise<GenerateImageResponse> {
    return apiPost<GenerateImageResponse>('/images/generate', { prompt, style })
  },

  /** List recently generated images from the gallery. */
  async gallery(): Promise<GalleryResponse> {
    return apiGet<GalleryResponse>('/images/gallery')
  },

  /** List available image generation styles. */
  async styles(): Promise<{ styles: [string, string][] }> {
    return apiGet<{ styles: [string, string][] }>('/images/styles')
  },
}
