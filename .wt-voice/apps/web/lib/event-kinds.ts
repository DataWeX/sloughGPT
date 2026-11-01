/**
 * Event kinds — dependency-free catalog for structured UI logging.
 *
 * This module has ZERO imports on purpose: `dev-log.ts` and `state-events.ts`
 * need the classifier at module scope, and dozens of test files mock
 * `@/lib/error-store`. Keeping the catalog here means those mocks can never
 * break module collection (no missing-export errors).
 *
 * Canonical envelope (see `lib/state-events.ts`):
 *   { id, kind, event, message, from?, to?, timestamp, data? }
 */

export type StateEventKind =
  | 'connection'
  | 'startup'
  | 'sse'
  | 'health'
  | 'overlay'
  | 'api'
  | 'auth'
  | 'chat'
  | 'training'
  | 'model'
  | 'system'
  | 'ui'

const STATE_EVENT_KINDS: ReadonlySet<string> = new Set([
  'connection',
  'startup',
  'sse',
  'health',
  'overlay',
  'api',
  'auth',
  'chat',
  'training',
  'model',
  'system',
  'ui',
])

export function isStateEventKind(v: unknown): v is StateEventKind {
  return typeof v === 'string' && STATE_EVENT_KINDS.has(v)
}

/**
 * Classify an event name into a StateEventKind by prefix.
 *
 *   connection: connection_*        startup: startup_*          sse: sse_*
 *   health:     health_*            overlay: overlay_*          api: api_*
 *   auth:       auth_*, workspace_* chat: session_*, stream_*, chat_*,
 *               conversion_*, regenerate_*, soul_*
 *   training:   training_*, checkpoint_*, dataset_*, knowledge_*
 *   model:      model_*             system: vm_*, shell_*, webhook_*,
 *               operation_*, download_*, error_lifecycle_*
 *   ui:         route_*, locale_*, theme_*, palette_*, mode_*, settings_*,
 *               feedback_*
 */
export function inferStateKind(event: string): StateEventKind {
  if (event.startsWith('api_')) return 'api'
  if (event.startsWith('connection_')) return 'connection'
  if (event.startsWith('startup_')) return 'startup'
  if (event.startsWith('overlay_')) return 'overlay'
  if (event.startsWith('sse_')) return 'sse'
  if (event.startsWith('health_')) return 'health'
  if (event.startsWith('auth_') || event.startsWith('workspace_')) return 'auth'
  if (
    event.startsWith('session_') ||
    event.startsWith('stream_') ||
    event.startsWith('chat_') ||
    event.startsWith('conversion_') ||
    event.startsWith('regenerate_') ||
    event.startsWith('soul_')
  )
    return 'chat'
  if (
    event.startsWith('training_') ||
    event.startsWith('checkpoint_') ||
    event.startsWith('dataset_') ||
    event.startsWith('knowledge_')
  )
    return 'training'
  if (event.startsWith('model_')) return 'model'
  if (event.startsWith('error_lifecycle_')) return 'system'
  if (
    event.startsWith('vm_') ||
    event.startsWith('shell_') ||
    event.startsWith('webhook_') ||
    event.startsWith('operation_') ||
    event.startsWith('download_')
  )
    return 'system'
  if (
    event.startsWith('route_') ||
    event.startsWith('locale_') ||
    event.startsWith('theme_') ||
    event.startsWith('palette_') ||
    event.startsWith('mode_') ||
    event.startsWith('settings_') ||
    event.startsWith('feedback_')
  )
    return 'ui'
  return 'system'
}
