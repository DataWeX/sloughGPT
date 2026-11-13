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
      // CommonJS/Node script files (postcss.config.js, tailwind.config.js,
      // codemod-*.mjs, …) legitimately use require/module/process/console;
      // the TS ruleset and browser env would flag them otherwise.
      files: ['**/*.js', '**/*.cjs', '**/*.mjs'],
      env: { node: true },
      rules: {
        '@typescript-eslint/no-require-imports': 'off',
        '@typescript-eslint/no-var-requires': 'off',
      },
    },
    {
      // Vitest test files legitimately carry `/// <reference types="vitest" />`
      // for global describe/it typings; removing it would break `tsc`.
      files: ['**/*.test.ts', '**/*.test.tsx', '**/*.spec.ts', '**/*.spec.tsx'],
      rules: {
        '@typescript-eslint/triple-slash-reference': 'off',
        // Ruling 2026-10-24 (card d715327c): mocks are not production
        // contracts — test files may cast (`vi.mocked(...) as any`,
        // fireEvent props). Strictness stays on prod code; the warning
        // budget is measured against non-test files only.
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
}
