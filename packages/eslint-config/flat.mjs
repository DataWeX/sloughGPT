/**
 * Flat-config (ESLint ≥ 9) factories — the ESLint-9 migration targets for
 * index.js / react.js (legacy .eslintrc format, still used by apps/web on its
 * own pinned eslint 8).
 *
 * WHY A FACTORY WITH ZERO `require()`s: npm installs this package as a
 * `file:` symlink, and CI runs an isolated `npm ci` inside each consuming
 * package. Node resolves requires from the REAL path (packages/eslint-config/),
 * where CI creates no node_modules — so any plugin require here would fail in
 * CI. The plugin deps hoist into the CONSUMER's tree instead; the consumer
 * resolves them and injects them as arguments.
 *
 * Legacy → flat mapping notes:
 * - `extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended']`
 *   → `js.configs.recommended` first, then the plugin's own ready-made
 *   `configs['flat/recommended']` (parser + plugin objects embedded), same
 *   precedence (later wins on rule conflicts).
 * - `ignorePatterns` → a standalone `{ignores}` object (global ignores).
 * - `overrides` → `files`-scoped config objects.
 */
export function makeTsConfig({ js, tseslint }) {
  return [
    { ignores: ['node_modules/**', 'dist/**', '.next/**', '**/*.d.ts'] },
    // eslint:recommended — first, exactly like the legacy extends order.
    { rules: js.configs.recommended.rules },
    // base (parser + plugin) + TS-file core-rule overrides + TS recommended.
    ...tseslint.configs['flat/recommended'],
    {
      // Legacy `rules:` block from index.js — highest precedence.
      rules: {
        '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/consistent-type-imports': ['error', { disallowTypeAnnotations: false }],
        'no-console': ['warn', { allow: ['warn', 'error'] }],
        // Prettier (semi:false) emits defensive leading `;` ASI guards that
        // this rule flags; lint-staged runs eslint --fix then prettier, so the
        // two fight forever. Stylistic conflict — prettier owns it.
        'no-extra-semi': 'off',
      },
    },
    {
      // Vitest test files legitimately carry `/// <reference types="vitest" />`
      // for global describe/it typings; removing it would break `tsc`.
      files: ['**/*.test.ts', '**/*.test.tsx', '**/*.spec.ts', '**/*.spec.tsx'],
      rules: { '@typescript-eslint/triple-slash-reference': 'off' },
    },
  ];
}

export function makeReactConfig({ js, tseslint, reactPlugin, hooksPlugin }) {
  return [
    ...makeTsConfig({ js, tseslint }),
    {
      // react/recommended's legacy parserOptions (ecmaFeatures.jsx etc.) —
      // merged on top of the TS base, as eslintrc extends did.
      ...(reactPlugin.configs.recommended.parserOptions
        ? { languageOptions: { parserOptions: reactPlugin.configs.recommended.parserOptions } }
        : {}),
      plugins: { react: reactPlugin, 'react-hooks': hooksPlugin },
      settings: { react: { version: 'detect' } },
      rules: {
        // plugin:react/recommended, then plugin:react-hooks/recommended
        // (legacy extends order), then the legacy react.js rules block.
        ...reactPlugin.configs.recommended.rules,
        ...hooksPlugin.configs.recommended.rules,
        'react/react-in-jsx-scope': 'off',
        'react/prop-types': 'off',
      },
    },
    {
      // Storybook CSF `render(args)` is invoked as a component body by the
      // preview wrapper, so hooks there attach to that wrapper component.
      files: ['**/*.stories.ts', '**/*.stories.tsx'],
      rules: { 'react-hooks/rules-of-hooks': 'off' },
    },
  ];
}
