/**
 * ESLint 9 flat config (replaces .eslintrc.json — eslint 9 requires flat).
 * Rules live in ../../eslint-config/flat.mjs; plugins resolve HERE because CI's
 * isolated `npm ci` in packages/sdk-ts/typescript-sdk hoists them into this
 * package's node_modules (file: dep of @sloughgpt/eslint-config).
 */
import js from '@eslint/js';
import tseslint from '@typescript-eslint/eslint-plugin';

import { makeTsConfig } from '../../eslint-config/flat.mjs';

export default makeTsConfig({ js, tseslint });
