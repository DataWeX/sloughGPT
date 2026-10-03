/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: [
    './index.js',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  settings: {
    react: { version: 'detect' },
  },
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    // Phase 6: legacy surface — keep errors for real bugs only
    '@typescript-eslint/consistent-type-imports': 'off',
    '@typescript-eslint/no-require-imports': 'off',
    '@typescript-eslint/triple-slash-reference': 'off',
    '@typescript-eslint/no-unused-expressions': 'off',
    '@typescript-eslint/no-empty': 'off',
    'no-empty': 'off',
    'no-extra-semi': 'off',
    'no-undef': 'off',
    'require-yield': 'off',
    'prefer-const': 'off',
    'no-constant-condition': 'off',
    'no-self-assign': 'off',
    'no-useless-escape': 'off',
    'no-control-regex': 'off',
    'react/no-unescaped-entities': 'off',
    'react-hooks/rules-of-hooks': 'off',
    'no-console': 'off',
    '@typescript-eslint/no-this-alias': 'off',
    '@typescript-eslint/no-misused-new': 'off',
    '@typescript-eslint/no-unsafe-function-type': 'off',
    '@typescript-eslint/no-wrapper-object-types': 'off',
    'no-new-wrappers': 'off',
  },
  env: {
    browser: true,
    node: true,
    es2022: true,
  },
  overrides: [
    {
      files: ['**/*.test.{ts,tsx}', '**/*.cy.{ts,tsx}', 'vitest-setup.ts', 'vite/**/*.ts'],
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-unused-vars': 'off',
      },
    },
  ],
}
