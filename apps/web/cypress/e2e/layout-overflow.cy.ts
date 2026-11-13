/**
 * Layout overflow audit - pages must never scroll horizontally at the document
 * level, and vertical scrollers (overflow-y: auto) must not carry their own
 * horizontal overflow (that is what clips section labels and shows a stray
 * scrollbar inside cards / dialogs).
 */

const ROUTES = ['/', '/datasets', '/models', '/training', '/chat', '/settings', '/knowledge']
const VIEWPORTS = [
  [1280, 800],
  [900, 720],
] as const

describe('Layout overflow audit', () => {
  let AUT: Window

  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
  })

  ROUTES.forEach((route) => {
    VIEWPORTS.forEach(([w, h]) => {
      it(`${route} does not overflow at ${w}x${h}`, () => {
        cy.viewport(w, h)
        cy.visit(route, { failOnStatusCode: false })
        cy.wait(1800)
        cy.window().then((win) => {
          AUT = win
          win.document.head.insertAdjacentHTML(
            'beforeend',
            '<style>[class*="9999"]{display:none !important}</style>',
          )
        })

        cy.then(() => {
          const doc = AUT.document
          const vw = AUT.innerWidth
          const scrollers: { sel: string; delta: number }[] = []

          const describe = (el: Element): string => {
            const id = el.id ? `#${el.id}` : ''
            const cls = (el.getAttribute('class') || '')
              .split(/\s+/)
              .filter(Boolean)
              .slice(0, 3)
              .join('.')
            return `${el.tagName.toLowerCase()}${id}${cls ? '.' + cls : ''}`
          }

          doc.querySelectorAll<HTMLElement>('body *').forEach((el) => {
            const cs = AUT.getComputedStyle(el)
            if (cs.display === 'none' || cs.visibility === 'hidden') return

            const cls = el.getAttribute('class') || ''
            const wantsXScroll = /\boverflow-x-(auto|scroll)\b/.test(cls)
            if (
              (cs.overflowY === 'auto' || cs.overflowY === 'scroll') &&
              !wantsXScroll &&
              el.scrollWidth - el.clientWidth > 1
            ) {
              scrollers.push({ sel: describe(el), delta: el.scrollWidth - el.clientWidth })
            }
          })

          const pageOverflow = doc.documentElement.scrollWidth - vw
          expect(
            pageOverflow,
            `document scrolls horizontally at ${route} (${vw}px wide)`,
          ).to.be.at.most(1)
          expect(
            scrollers,
            `vertical scrollers with horizontal overflow on ${route}`,
          ).to.have.length(0)
        })
      })
    })
  })
})
