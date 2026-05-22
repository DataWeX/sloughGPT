export interface NetworkInfo {
  type: 'wifi' | 'cellular' | 'ethernet' | 'unknown'
  metered: boolean
  effectiveType: string | null
  downlink: number | null
}

export function getNetworkInfo(): NetworkInfo {
  const conn = (navigator as any).connection
  if (!conn) return { type: 'unknown', metered: false, effectiveType: null, downlink: null }

  let type: NetworkInfo['type'] = 'unknown'
  if (conn.type === 'wifi' || conn.type === 'ethernet') type = conn.type
  else if (conn.type === 'cellular') type = 'cellular'
  else if (conn.effectiveType === '4g' || conn.effectiveType === '3g') type = 'cellular'
  else if (conn.effectiveType === 'slow-2g' || conn.effectiveType === '2g') type = 'cellular'

  const effectiveType: string | null = conn.effectiveType || null
  const downlink: number | null = typeof conn.downlink === 'number' ? conn.downlink : null

  // Metered: explicit flag, or cellular, or very slow
  const metered = !!(conn.saveData || conn.type === 'cellular' || (downlink !== null && downlink < 0.5))

  return { type, metered, effectiveType, downlink }
}

export function isMeteredConnection(): boolean {
  return getNetworkInfo().metered
}

export function connectionLabel(info: NetworkInfo): string {
  if (info.type === 'cellular') {
    return `${info.effectiveType?.toUpperCase() || 'cellular'} ${info.downlink ? `(${info.downlink.toFixed(1)} Mbps)` : ''}`
  }
  if (info.type === 'wifi') return `WiFi ${info.downlink ? `(${info.downlink.toFixed(0)} Mbps)` : ''}`
  if (info.type === 'ethernet') return `Ethernet`
  return info.downlink ? `${info.downlink.toFixed(0)} Mbps` : 'unknown'
}
