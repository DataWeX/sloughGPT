/**
 * Datasets page - import, list, export, delete datasets
 */

const API = 'http://localhost:8000'

describe('Datasets page', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/datasets')
  })

  it('renders the page header', () => {
    cy.contains('h1', 'Datasets').should('be.visible')
  })

  it('shows Import button', () => {
    cy.contains('button', /^Import$/).should('be.visible')
  })

  it('shows Refresh button', () => {
    cy.contains('button', 'Refresh').should('be.visible')
  })

  it('shows empty state when no datasets', () => {
    cy.contains('No datasets yet').should('be.visible')
    cy.contains('button', 'Import Dataset').should('be.visible')
  })

  it('opens the Import Dataset modal', () => {
    cy.contains('button', /^Import$/).click()
    cy.contains('Import Dataset').should('be.visible')
    cy.get('input[placeholder="Dataset name (optional)"]').should('exist')
  })
})

describe('Dataset Import Modal', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.visit('/datasets')
    cy.contains('button', /^Import$/).click()
  })

  it('defaults to Local Path source', () => {
    cy.get('[role="dialog"]').contains('button', 'Local Path').should('be.visible')
    cy.get('input[placeholder="/path/to/dataset/folder"]').should('be.visible')
  })

  it('shows all source tabs', () => {
    cy.get('[role="dialog"]').contains('button', 'GitHub').should('be.visible')
    cy.get('[role="dialog"]').contains('button', 'HuggingFace').should('be.visible')
    cy.get('[role="dialog"]').contains('button', 'URL').should('be.visible')
  })

  it('switches to GitHub source', () => {
    cy.get('[role="dialog"]').contains('button', 'GitHub').click()
    cy.get('input[placeholder="https://github.com/user/repo"]').should('be.visible')
  })

  it('switches to HuggingFace source', () => {
    cy.get('[role="dialog"]').contains('button', 'HuggingFace').click()
    cy.get('input[placeholder="username/dataset-name"]').should('be.visible')
  })

  it('switches to URL source', () => {
    cy.get('[role="dialog"]').contains('button', 'URL').click()
    cy.get('input[placeholder="https://example.com/data.txt"]').should('be.visible')
  })

  it('shows dataset name input', () => {
    cy.get('input[placeholder="Dataset name (optional)"]').should('be.visible')
  })

  it('has an Import action', () => {
    cy.get('[role="dialog"]')
      .contains('button', /^Import$/)
      .should('be.visible')
  })
})

describe('Dataset Import Modal - layout', () => {
  let AUT: Window

  const VIEWPORTS = [
    [1280, 800],
    [900, 720],
    [700, 720],
    [640, 720],
    [548, 720],
    [500, 700],
  ] as const

  beforeEach(() => {
    cy.on('uncaught:exception', () => false)
    cy.mockAll()
  })

  VIEWPORTS.forEach(([w, h]) => {
    it(`is centered and overflow-free at ${w}x${h}`, () => {
      cy.viewport(w, h)
      cy.visit('/datasets')
      cy.wait(3000)
      cy.window().then((win) => {
        AUT = win
        win.document.head.insertAdjacentHTML(
          'beforeend',
          '<style>[class*="9999"]{display:none !important}</style>',
        )
      })
      cy.contains('button', /^(Import|Add file)$/)
        .first()
        .click({ force: true })
      cy.get('[role="dialog"]').should('contain', 'Import Dataset')
      cy.contains('button', 'GitHub').first().click({ force: true })

      const measure = () => {
        const doc = AUT.document
        const label = doc.querySelector('label[for="github-search"]') as HTMLElement
        const input = doc.querySelector('#github-search') as HTMLElement
        const dlg = label.closest('[role="dialog"]') as HTMLElement
        const title = dlg.querySelector('h2') as HTMLElement
        const inner = label.parentElement!.parentElement!.parentElement as HTMLElement
        const r = (el: HTMLElement) => el.getBoundingClientRect()
        const d = r(dlg)
        const ir = r(inner)

        const culprits: string[] = []
        inner.querySelectorAll('*').forEach((n) => {
          if (n.classList.contains('sr-only') || n.closest('.sr-only')) return
          const b = (n as HTMLElement).getBoundingClientRect()
          if (b.width > 0 && (b.right > ir.right + 0.5 || b.left < ir.left - 0.5)) {
            culprits.push(
              `${n.tagName}.${(n.getAttribute('class') || '').slice(0, 30)}:${(n.textContent || '').slice(0, 20)}`,
            )
          }
        })

        return {
          d,
          expectX: (AUT.innerWidth - d.width) / 2,
          expectY: (AUT.innerHeight - d.height) / 2,
          pageOverflow: doc.documentElement.scrollWidth - AUT.innerWidth,
          innerDelta: inner.scrollWidth - inner.clientWidth,
          innerSl: inner.scrollLeft,
          labelX: r(label).x,
          inputX: r(input).x,
          titleX: r(title).x,
          inBody: dlg.parentElement === doc.body,
          culprits,
          vw: AUT.innerWidth,
        }
      }

      cy.get('label[for="github-search"]', { timeout: 15000 }).should(() => {
        const m = measure()
        expect(
          Math.abs(m.d.x - m.expectX),
          `dialog x centered (${m.d.x} vs ${m.expectX})`,
        ).to.be.lessThan(2)
        expect(
          Math.abs(m.d.y - m.expectY),
          `dialog y centered (${m.d.y} vs ${m.expectY})`,
        ).to.be.lessThan(2)
        expect(m.d.x, 'dialog left >= 0').to.be.gte(0)
        expect(m.d.right, 'dialog right <= viewport').to.be.at.most(m.vw + 0.5)
        expect(m.pageOverflow, 'no page-level horizontal overflow').to.be.at.most(1)
        expect(m.innerDelta, 'no horizontal overflow inside dialog scroller').to.be.at.most(1)
        expect(m.innerSl, 'dialog scroller is not scrolled horizontally').to.be.eq(0)
        expect(m.labelX, 'label aligned with input').to.be.closeTo(m.inputX, 0.5)
        expect(m.labelX, 'label aligned with title').to.be.closeTo(m.titleX, 0.5)
        expect(m.culprits, 'no overflowing elements').to.have.length(0)
        expect(m.inBody, 'dialog is portaled to body').to.be.true
      })
    })
  })
})

describe('Dataset cards', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.intercept('GET', `${API}/datasets`, {
      statusCode: 200,
      body: {
        datasets: [
          {
            id: 'ds1',
            name: 'shakespeare',
            source: 'local',
            size: 1048576,
            samples: 500,
            created_at: '2026-08-01T00:00:00Z',
          },
          {
            id: 'ds2',
            name: 'tinyshakespeare',
            source: 'huggingface',
            size: 2097152,
            samples: 1200,
            created_at: '2026-08-02T00:00:00Z',
          },
        ],
      },
    }).as('datasetsList')
    cy.intercept('GET', `${API}/datasets/*/versions`, {
      statusCode: 200,
      body: { versions: [], count: 0 },
    }).as('versions')
    cy.visit('/datasets')
  })

  it('shows dataset names', () => {
    cy.contains('shakespeare').should('be.visible')
    cy.contains('tinyshakespeare').should('be.visible')
  })

  it('shows Export buttons on dataset cards', () => {
    cy.get('button[aria-label^="Export"]').should('have.length', 2)
  })

  it('shows Train buttons on dataset cards', () => {
    cy.get('button[aria-label^="Train with"]').should('have.length', 2)
  })
})
