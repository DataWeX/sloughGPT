export interface ApiDocEndpoint {
  method: string
  path: string
  description: string
  body?: { field: string; type: string; required: boolean }[]
}

export type ApiDocBodyField = NonNullable<ApiDocEndpoint['body']>[number]
