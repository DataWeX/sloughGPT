/**
 * Search timing constant and the pure topic-suggestion heuristic for the
 * Knowledge add dialog. No React, no IO.
 */
export const SEARCH_DEBOUNCE_MS = 300

export const suggestTopic = (content: string): string => {
  const lower = content.toLowerCase()
  if (/\b(prefer|like|love|hate|favorite|best|worst)\b/.test(lower)) return 'preferences'
  if (/\b(born|live|work|study|age|name|from)\b/.test(lower)) return 'personal'
  if (/\b(project|code|api|bug|fix|deploy|build)\b/.test(lower)) return 'technical'
  if (/\b(meeting|deadline|schedule|plan|goal)\b/.test(lower)) return 'planning'
  if (/\b(book|read|watch|listen|play|game|movie)\b/.test(lower)) return 'interests'
  if (/\b(cook|recipe|food|eat|drink|coffee|tea)\b/.test(lower)) return 'food'
  return 'general'
}
