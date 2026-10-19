/**
 * Canaries for the chat-layout shell tokens (card 0f4d548a).
 *
 * Two failure modes this guards against, both of which are silent in the
 * browser — no build error, no console noise, just layout rot:
 *
 * 1. Deleting `--nav-tab-protrusion` from globals.css: both consumers
 *    (AppLayout's bookmark tab, ConversationSidebar's docked gutter) resolve
 *    `var()` to nothing → tab width/gutter padding drop to initial values and
 *    the tab collides with conversation row titles again (the bug fixed by
 *    8aec74365 / card 88e4d9fb).
 * 2. Re-introducing a magic `1.15rem` in one consumer only: the tab and the
 *    gutter drift apart — the sticker stops landing inside the notch.
 *
 * Runs in the node environment (lib/** per vitest.config.ts) so fs reads of
 * the source tree work; features/** tests are jsdom and must not do this.
 */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const read = (rel: string) => readFileSync(join(webRoot, rel), 'utf8')

describe('chat layout shell tokens', () => {
  const css = read('app/globals.css')

  it('defines --nav-tab-protrusion (tab width and conv-sidebar gutter share it)', () => {
    expect(css).toMatch(/--nav-tab-protrusion:\s*1\.15rem/)
  })

  it('pins the conversation column to full width (chip cap must not creep back)', () => {
    expect(css).toMatch(/--chat-thread-max:\s*100%/)
  })

  it('both consumers read the shared token — no magic 1.15rem on either side', () => {
    const appLayout = read('components/AppLayout.tsx')
    const convSidebar = read('features/chat/components/sidebar/ConversationSidebar.tsx')

    expect(appLayout).toContain('var(--nav-tab-protrusion)')
    expect(convSidebar).toContain('var(--nav-tab-protrusion)')
    expect(appLayout).not.toContain('1.15rem')
    expect(convSidebar).not.toContain('1.15rem')
  })

  it('chat column wrappers no longer cap the thread at max-w-4xl', () => {
    const capped = [
      'features/chat/components/layout/ChatScreen.tsx',
      'features/chat/components/input/ChatInput.tsx',
      'features/chat/components/messages/ReasoningPanel.tsx',
    ]
    for (const file of capped) {
      expect(read(file), `${file} must not re-cap the column`).not.toContain('max-w-4xl')
    }
  })
})
