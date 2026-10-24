/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ['./index.js', 'next/core-web-vitals'],
  // eslint-config-next injects its own parser (next/dist/compiled/babel/…),
  // which does not forward @typescript-eslint parser services — the base
  // chain's consistent-type-imports rule then crashes at rule load with
  // "requires type information". Pin the TS parser (own config beats extends):
  // it provides services and handles .tsx natively, same as strui/sdk.
  parser: '@typescript-eslint/parser',
  rules: {
    '@next/next/no-img-element': 'off',
  },
  settings: {
    // Monorepo: lint-staged runs eslint from the repo root, so tell the
    // Next plugin where the app lives instead of relying on CWD.
    next: {
      rootDir: ['apps/web/'],
    },
  },
}
