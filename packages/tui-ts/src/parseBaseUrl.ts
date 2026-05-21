/**
 * Optional `--url` CLI flag (first match wins). Returns base URL without trailing slash.
 */
export function parseBaseUrl(argv: string[]): string | undefined {
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--url' && args[i + 1]) {
      return args[i + 1].replace(/\/$/, '');
    }
    if (args[i].startsWith('--url=')) {
      return args[i].slice('--url='.length).replace(/\/$/, '');
    }
  }
  return undefined;
}
