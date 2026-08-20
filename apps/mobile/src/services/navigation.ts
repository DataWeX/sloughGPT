/**
 * Navigation service — allows navigating from outside React components
 * (push notifications, deep links, etc.)
 */

type NavigateFn = (screen: string) => void;

let _navigate: NavigateFn | null = null;

export function setNavigationRef(fn: NavigateFn) {
  _navigate = fn;
}

export function navigateTo(screen: string) {
  if (_navigate) {
    _navigate(screen);
  }
}
