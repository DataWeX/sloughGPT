import { describe, it, expect } from 'vitest'
import { REDIRECTS } from './redirects'
import { ALL_ROUTES } from './navigation'
import { routePathList } from '../vite/routes'
import contract from './route-contract.json'

/**
 * CONTENT pins for the two hand-maintained route lists.
 *
 * Everything else around these lists pins only *shape*: `navigation.test.ts`
 * asserts paths are unique and labelKeys match a pattern, and the redirect
 * tests pin a COUNT (46). Those all stay green when a route is added,
 * deleted or retargeted — which is how `/files` ended up as a redirect
 * source while still being a live page: the count never moved, so nothing
 * objected and the 307 fired before routing made the page unreachable.
 *
 * This file pins the actual CONTENT instead, in both directions. A route
 * change — even one-for-one, so the count is unchanged — now fails here
 * until `lib/route-contract.json` is edited deliberately in the same commit.
 *
 * The contract file is hand-maintained, never generated: a pin derived from
 * what it pins verifies nothing.
 */

const HINT =
  'route change intentional? edit lib/route-contract.json in the same commit and explain why.'

describe('route contract — content pins', () => {
  it('redirect table matches the contract exactly', () => {
    // toEqual is bidirectional: catches a redirect added, removed, retargeted,
    // or swapped for another while keeping the total at 46.
    expect(REDIRECTS, HINT).toEqual(contract.redirects)
  })

  it('nav routes match the contract exactly, in sidebar order', () => {
    expect(
      ALL_ROUTES.map((r) => r.path),
      HINT,
    ).toEqual(contract.navRoutes)
  })

  it('no NEW page is ever shadowed by a redirect', () => {
    // Regression guard for a real shipped bug: `/files` was a redirect source
    // while still being a live page, and applyRedirect fires BEFORE routing —
    // so the 307 made the page unreachable. Checking only nav routes would
    // miss it (that page is not in nav), so the live set is every page on
    // disk, derived from the filesystem rather than from either list.
    //
    // `knownShadowed` records pages ALREADY in this state (empty since
    // 2026-10-24 — all 36 were deleted, page.tsx 109 -> 73). It may only
    // shrink: if you delete a dead page, drop its entry in the same commit;
    // if this test fails you have just created an unreachable page.
    const live = new Set(routePathList)
    const shadowed = Object.keys(REDIRECTS)
      .filter((source) => live.has(source))
      .sort()
    expect(shadowed, HINT).toEqual([...contract.knownShadowed].sort())
  })

  it('no NEW route joins a concept-namespaced prefix', () => {
    // Ruling: "workspace is a philosophy, not a page." A URL namespace is an
    // address space (redirects, bookmarks, scopes and tests depend on it);
    // a concept is a taxonomy. A taxonomy must not own an address space,
    // because when the concept is renamed the URLs cannot follow.
    //
    // Workspace is a SCOPE — you are IN one and everything else is scoped by
    // it — so it belongs in a selector and a nav grouping LABEL, not a path
    // prefix. The 4 routes already nested there are recorded so they may only
    // shrink; a failure here means a new concept-namespaced route was added.
    const namespaced = contract.navRoutes
      .filter((path) => path === '/workspace' || path.startsWith('/workspace/'))
      .sort()
    expect(namespaced, HINT).toEqual([...contract.knownWorkspaceNamespace].sort())
  })

  it('contract navRoutes are unique and route-shaped', () => {
    expect(contract.navRoutes.length).toBe(new Set(contract.navRoutes).size)
    for (const path of contract.navRoutes) {
      expect(path).toMatch(/^\/[a-z0-9/[\]-]*$/i)
    }
  })

  it('every redirect moves between route-shaped paths', () => {
    for (const [source, target] of Object.entries(contract.redirects)) {
      expect(source, `source ${source}`).toMatch(/^\//)
      expect(target, `target of ${source} -> ${target}`).toMatch(/^\//)
    }
  })

  it('every nav route resolves to a real page', () => {
    // routePathList is derived from the filesystem (vite/routes.ts), so this
    // is the nav -> surface binding: a nav entry with no page.tsx behind it
    // is a link to a 404.
    const real = new Set(routePathList)
    const dead = contract.navRoutes.filter((path) => !real.has(path))
    expect(dead, `nav links to nothing: ${dead.join(', ')}`).toEqual([])
  })

  it('every redirect target resolves to a real page', () => {
    // Same binding on the redirect side: send users to a surface that exists.
    const real = new Set(routePathList)
    const dead = Object.entries(contract.redirects)
      .map(([source, target]) => [source, target.split('?')[0] ?? ''] as const)
      .filter(([, targetPath]) => !real.has(targetPath))
      .map(([source, target]) => `${source} -> ${target}`)
    expect(dead, `redirect targets a missing page: ${dead.join(', ')}`).toEqual([])
  })
})
