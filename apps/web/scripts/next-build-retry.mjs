/**
 * Runs `next build` once; on failure waits 2s and retries once.
 * Mitigates intermittent ENOENT / manifest races seen after `rm -rf .next`.
 */
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

function runBuild() {
  return spawnSync('npx', ['next', 'build'], {
    stdio: 'inherit',
    cwd: root,
    shell: true,
    env: process.env,
  })
}

let result = runBuild()
if (result.status === 0) {
  process.exit(0)
}

console.warn('[next-build-retry] first build failed (exit %s); retrying once in 2s…', result.status ?? 'unknown')
await new Promise((resolve) => setTimeout(resolve, 2000))

result = runBuild()
process.exit(result.status ?? 1)
