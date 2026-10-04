// ApiError — the single error model shared by every HTTP path.

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: {
      raw?: string
      code?: string
      details?: unknown
      correlationId?: string
    },
    public requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}
