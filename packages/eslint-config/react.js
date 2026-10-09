/** @type {import('eslint').Linter.Config} */
module.exports = {
  extends: ['./index.js', 'plugin:react/recommended', 'plugin:react-hooks/recommended'],
  settings: {
    react: { version: 'detect' },
  },
  rules: {
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
  },
  overrides: [
    {
      // Storybook CSF `render(args)` is invoked as a component body by the
      // preview wrapper, so hooks there attach to that wrapper component.
      files: ['**/*.stories.ts', '**/*.stories.tsx'],
      rules: { 'react-hooks/rules-of-hooks': 'off' },
    },
  ],
}
