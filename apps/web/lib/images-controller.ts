import { apiPost } from '@/lib/http-client'

export interface GenerateImageResponse {
  image: string
  style: string
  prompt: string
  id: string
}

export interface GalleryImageInfo {
  id: string
  path: string
  created: number
}

export interface GalleryResponse {
  images: GalleryImageInfo[]
}

export interface StylesResponse {
  styles: [string, string][]
}

export const imagesController = {
  async generate(prompt: string, style: string): Promise<GenerateImageResponse> {
    return apiPost<GenerateImageResponse>('/images/generate', { prompt, style })
  },

  async listGallery(): Promise<GalleryResponse> {
    return fetch('/images/gallery').then(r => r.json())
  },

  async listStyles(): Promise<StylesResponse> {
    return fetch('/images/styles').then(r => r.json())
  },
}
