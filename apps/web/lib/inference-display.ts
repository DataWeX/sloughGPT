/**
 * Whether a catalog model id corresponds to the API's currently loaded `model_type`
 * from `GET /health` (ids may be short names or path-like).
 */
export function catalogIdMatchesRuntime(catalogId: string, runtimeType: string): boolean {
  const c = catalogId.trim().toLowerCase()
  const r = runtimeType.trim().toLowerCase()
  if (!c || !r) return false
  if (c === r) return true
  const cLast = c.split('/').pop() ?? c
  const rLast = r.split('/').pop() ?? r
  if (cLast === rLast) return true
  if (c.endsWith(`/${r}`) || c.endsWith(`/${rLast}`)) return true
  if (r.endsWith(`/${c}`) || r.endsWith(`/${cLast}`)) return true
  return false
}
