/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: [
    './index.js',
    'next/core-web-vitals',
  ],
  rules: {
    '@next/next/no-img-element': 'off',
  },
}
