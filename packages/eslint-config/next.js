/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ['./index.js', 'next/core-web-vitals'],
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
