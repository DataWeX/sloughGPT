import { type OfflineCanvasOptions } from 'next/dist/server/use-cache/cache-manifest';

declare module 'next/dist/server/use-cache/cache-manifest' {
  export interface OfflineCanvasOptions {
    preferOffline?: boolean;
  }
}

export {};
