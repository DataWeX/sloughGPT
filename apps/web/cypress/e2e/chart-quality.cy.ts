/**
 * Chart quality — shared chart layer journey assertions (card 96d1507c, slice d).
 *
 * Zero-flake floor of the sizing report's canary plan
 * (docs/plans/2026-10-22-chart-quality-sizing.md §6.1): DOM/state assertions
 * that run in the existing frontend-e2e CI job. The pixel-diff tier is the
 * second step and needs committed baselines from a browserful machine.
 *
 * Under cy.mockAll() there is no SSE history, so the monitoring charts
 * deterministically sit in their empty states — the exact "silent blank"
 * defects this card removed now render a labeled empty state instead.
 */
describe('Chart quality — frame states across the estate', () => {
  describe('/monitoring charts', () => {
    beforeEach(() => {
      cy.mockAll()
      cy.visit('/monitoring')
    })

    it('shows the trend card even before data loads (no loaded&& blank)', () => {
      cy.contains('Health & memory trend').should('be.visible')
      cy.get('[data-testid="chart-frame"]').should('exist')
    })

    it('renders the trend empty state until history builds', () => {
      cy.contains('No trend data yet').should('be.visible')
      cy.get('[data-testid="chart-frame"][data-state="empty"]').should('exist')
    })

    it('shows the real-time chart card with its empty state (no >1-point gate)', () => {
      // The Diagnostics fold is native <details> — children stay mounted.
      cy.contains('Real-time chart').should('exist')
      cy.contains('Waiting for data...').should('exist')
    })
  })

  describe('/multimodal accuracy chart', () => {
    beforeEach(() => {
      cy.mockAll()
      cy.visit('/multimodal')
    })

    it('shows the accuracy chart empty state with zero learned images', () => {
      cy.contains('Accuracy history builds as training runs').should('exist')
    })
  })

  describe('/training job loss chart', () => {
    beforeEach(() => {
      cy.mockAll()
    })

    it('routes the loss card through the shared frame when the job renders', () => {
      cy.visit('/training/job/test-job-id')
      // Same defensive pattern as training-job-detail-page.cy.ts: the page
      // shows either the job view or its not-found state under mocks.
      cy.get('body').then(($body) => {
        if ($body.text().includes('Not Found')) return
        cy.get('[data-testid="chart-frame"]').should('exist')
      })
    })
  })
})
