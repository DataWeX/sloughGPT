import { type NextRequest } from '@/vite/next-compat/server'
import { proxyRequest } from '@/lib/planner-proxy'

export async function POST(request: NextRequest) {
  const body = await request.json()
  return proxyRequest('/api/board/transition', {
    method: 'POST',
    body: JSON.stringify(body),
    // first-party relay marker consumed by proxyRequest (card 20261108_142919)
    headers: { 'x-slo-relay': '1' },
  })
}
