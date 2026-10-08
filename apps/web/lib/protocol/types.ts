// Protocol schema vocabulary — the frontend projection of the backend's
// descriptor contract (ToolSpec → registry row, see docs/TRANSPORT_PROJECTIONS.md).
// The descriptor types live in the GENERATED file so there is exactly one
// definition: `contracts.gen.ts` declares them, this module re-exports them for
// consumers that want the vocabulary without pulling in the CONTRACTS table.

export type { ContractDescriptor, JsonSchema } from './contracts.gen'

export type ReadStatus = 'idle' | 'loading' | 'success' | 'error'

/** Query parameters for a capability read — coerced to strings on the wire. */
export type EndpointParams = Record<string, string | number | boolean>

/** One read's state machine slice, keyed by capability + version + params. */
export interface ReadSlice {
  status: ReadStatus
  data?: unknown
  error?: string
  /** Epoch ms of the last settled attempt (0 while idle). */
  updatedAt: number
}
