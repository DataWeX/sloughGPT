/**
 * Type declarations for custom Cypress commands (api-mocks.ts)
 */
declare namespace Cypress {
  interface Chainable<Subject = any> {
    /** Mock /health endpoint */
    mockHealth(overrides?: Record<string, unknown>): Chainable<null>

    /** Mock /models endpoint */
    mockModels(): Chainable<null>

    /** Mock /datasets endpoint */
    mockDatasets(): Chainable<null>

    /** Mock /system/* and /health/detailed endpoints */
    mockSystem(): Chainable<null>

    /** Mock /tokenizer/* endpoints */
    mockTokenizer(): Chainable<null>

    /** Mock /knowledge CRUD endpoints */
    mockKnowledge(items?: string[]): Chainable<null>

    /** Mock all endpoints via individual mocks */
    mockAll(): Chainable<null>
  }
}
