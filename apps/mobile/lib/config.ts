import Constants from 'expo-constants'

const DEFAULT_API_URL = 'http://localhost:8000'

export function getApiUrl(): string {
  const extra = Constants.expoConfig?.extra
  if (extra?.apiUrl) return extra.apiUrl

  const debuggerHost = Constants.expoGoConfig?.debuggerHost
  if (debuggerHost) {
    const host = debuggerHost.split(':')[0]
    return `http://${host}:8000`
  }

  return DEFAULT_API_URL
}

export const API_URL = getApiUrl()
