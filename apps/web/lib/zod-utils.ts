import type { ZodError, ZodSchema } from 'zod'

/**
 * Extract Zod validation errors into a flat field → message map.
 *
 * Usage:
 *   const result = schema.safeParse(data)
 *   if (!result.success) {
 *     const errors = extractFieldErrors(result.error, { name: 'name', desc: 'description' })
 *     setErrors(errors)
 *     return
 *   }
 *
 * @param error - ZodError from a failed safeParse
 * @param fieldMap - map from your local field names to Zod path strings
 * @returns object with field names as keys and error messages as values
 */
export function extractFieldErrors<T extends string>(
  error: ZodError,
  fieldMap: Record<T, string>,
): Partial<Record<T, string>> {
  const result: Partial<Record<T, string>> = {}
  for (const issue of error.issues) {
    const path = issue.path[0] as string
    for (const [localName, zodPath] of Object.entries(fieldMap)) {
      if (path === zodPath) {
        result[localName as T] = issue.message
      }
    }
  }
  return result
}
