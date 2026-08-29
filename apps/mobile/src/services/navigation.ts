/**
 * Navigation service — allows navigating from outside React components
 * (push notifications, deep links, etc.)
 */

type NavigateFn = (screen: string) => void;

let _navigate: NavigateFn | null = null;

export function setNavigationRef(fn: NavigateFn) {
  _navigate = fn;
}

/** Top-level screens the sidebar context understands. */
const TOP_LEVEL = new Set(['Home', 'Chat', 'Models', 'Tools', 'Settings']);

/** Map deep link paths / notification screen names → sidebar screen. */
const SCREEN_MAP: Record<string, string> = {
  // top-level passthrough
  home: 'Home', chat: 'Chat', models: 'Models', tools: 'Tools', settings: 'Settings',
  // tools sub-screens → land on Tools
  training: 'Tools', knowledge: 'Tools', health: 'Tools', souls: 'Tools',
  datasets: 'Tools', benchmark: 'Tools', feedback: 'Tools', workflow: 'Tools',
  voice: 'Tools', companion: 'Tools', learn: 'Tools', agents: 'Tools',
  multimodal: 'Tools', export: 'Tools', 'import': 'Tools', tokenizer: 'Tools',
  compare: 'Tools', adapters: 'Tools', errors: 'Tools', security: 'Tools',
  images: 'Tools', experiments: 'Tools', files: 'Tools', registry: 'Tools',
  memory: 'Tools',
};

export function navigateTo(raw: string) {
  if (!_navigate) return;

  // Strip scheme / host prefix
  const path = raw
    .replace(/^sloughgpt:\/\//, '')
    .replace(/^https?:\/\/sloughgpt\.app\//, '')
    .replace(/^\/+/, '');

  if (!path) { _navigate('Chat'); return; }

  // "tools/training" → Tools, "settings/health" → Settings
  const segments = path.split('/');
  const topSegment = segments[0].toLowerCase();
  const target = SCREEN_MAP[topSegment];

  if (target) {
    _navigate(target);
  } else if (TOP_LEVEL.has(segments[0])) {
    _navigate(segments[0]);
  } else {
    _navigate('Chat');
  }
}
