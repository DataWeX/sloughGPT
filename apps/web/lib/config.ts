/** FastAPI backend base URL for the Next.js app (client bundles). */
export const PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/** API endpoint path - use direct backend URL to avoid proxy issues with streaming */
export const API_CHAT_ENDPOINT = `${PUBLIC_API_URL}/chat/stream`

/** localStorage key for user-injected knowledge snippets */
export const KNOWLEDGE_STORAGE_KEY = 'man_injected_knowledge'
