import { describe, expect, it } from 'vitest';
import { parseBaseUrl } from './parseBaseUrl.js';

describe('parseBaseUrl', () => {
  it('parses --url value', () => {
    expect(
      parseBaseUrl(['node', 'cli', '--url', 'http://localhost:9000/'])
    ).toBe('http://localhost:9000');
  });

  it('parses --url=value', () => {
    expect(parseBaseUrl(['node', 'cli', '--url=http://127.0.0.1:8000'])).toBe(
      'http://127.0.0.1:8000'
    );
  });

  it('returns undefined when absent', () => {
    expect(parseBaseUrl(['node', 'cli'])).toBeUndefined();
  });
});
