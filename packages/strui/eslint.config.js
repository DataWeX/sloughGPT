/**
 * ESLint 9 flat config (replaces .eslintrc.json — eslint 9 requires flat).
 * Rules live in ../eslint-config/flat.mjs; plugins resolve HERE because CI's
 * isolated `npm ci` in packages/strui hoists them into this package's
 * node_modules (file: dep of @sloughgpt/eslint-config).
 */
import js from '@eslint/js';
import tseslint from '@typescript-eslint/eslint-plugin';
import reactPlugin from 'eslint-plugin-react';
import hooksPlugin from 'eslint-plugin-react-hooks';

import { makeReactConfig } from '../eslint-config/flat.mjs';

export default makeReactConfig({ js, tseslint, reactPlugin, hooksPlugin });
