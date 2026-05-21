#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import App from './App.js';
import { parseBaseUrl } from './parseBaseUrl.js';

const fromFlag = parseBaseUrl(process.argv);
const fromEnv = process.env.SLOUGHGPT_API_URL?.replace(/\/$/, '');
const baseUrl = fromFlag ?? fromEnv ?? 'http://127.0.0.1:8000';

render(<App baseUrl={baseUrl} />);
