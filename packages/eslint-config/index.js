/** @type {import('eslint').Linter.Config} */
module.exports = {
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended'],
  rules: {
    '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/consistent-type-imports': ['error', { disallowTypeAnnotations: false }],
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    // Prettier (semi:false) emits defensive leading `;` ASI guards that this
    // rule flags; lint-staged runs eslint --fix then prettier, so the two
    // fight forever. Stylistic conflict — prettier owns it.
    'no-extra-semi': 'off',
  },
  ignorePatterns: ['node_modules/', 'dist/', '.next/', '*.d.ts'],
  overrides: [
    {
      // Vitest test files legitimately carry `/// <reference types="vitest" />`
      // for global describe/it typings; removing it would break `tsc`.
      files: ['**/*.test.ts', '**/*.test.tsx', '**/*.spec.ts', '**/*.spec.tsx'],
      rules: { '@typescript-eslint/triple-slash-reference': 'off' },
    },
  ],
}
