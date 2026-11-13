'use client'

import type { HealthStatus as ApiHealth } from '@/lib/model-controller'

export type ApiHealthSnapshot = ApiHealth | 'offline' | null
