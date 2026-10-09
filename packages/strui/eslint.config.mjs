// ESLint 9 reads flat config only, but this package's rules live in the shared
// @sloughgpt/eslint-config in eslintrc format: apps/web and apps/lib are still
// on ESLint 8 (`next lint`) and reach those rules through `extends`. Breaking
// the eslintrc export to satisfy v9 would strand them, and restating the rules
// here would fork the shared copy into a second source of truth.
//
// FlatCompat is ESLint's supported bridge, so .eslintrc.json stays the single
// declaration and this file only translates its shape.
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'
import { FlatCompat } from '@eslint/eslintrc'
import js from '@eslint/js'

const compat = new FlatCompat({
  baseDirectory: dirname(fileURLToPath(import.meta.url)),
  // The shared config extends `eslint:recommended`, which FlatCompat cannot
  // resolve on its own — v9 moved that preset into @eslint/js and it must be
  // handed over explicitly.
  recommendedConfig: js.configs.recommended,
})

export default [...compat.extends('./.eslintrc.json')]
