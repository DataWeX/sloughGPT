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

    /** Mock /knowledge CRUD endpoints */
    mockKnowledge(items?: string[]): Chainable<null>

    /** Mock all endpoints via individual mocks */
    mockAll(): Chainable<null>

    /** Mock VLM checkpoints, status, DPO, and train endpoints */
    mockVisual(): Chainable<null>

    /** Mock multimodal capabilities, training report, and reset */
    mockMultimodal(): Chainable<null>

    /** Mock /agents CRUD and execute endpoints */
    mockAgents(overrides?: any[]): Chainable<null>

    /** Mock /vm/run, /vm/builtins, /vm/info, and training job endpoints */
    mockVm(runOverrides?: Record<string, unknown>): Chainable<null>
  }
}
