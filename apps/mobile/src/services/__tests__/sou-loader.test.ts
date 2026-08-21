jest.mock('../../services/api-client', () => ({
  getApiUrl: jest.fn().mockResolvedValue('http://localhost:8000'),
}));

jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

let mockPickDocument: jest.Mock;
jest.mock('expo-document-picker', () => ({
  getDocumentAsync: (...args: any[]) => mockPickDocument(...args),
}));

import * as souLoader from '../sou-loader';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Build a minimal valid .sou v3 binary buffer
function buildMinimalSou(
  metadata: Record<string, unknown>,
  weights: Record<string, Float32Array>,
): ArrayBuffer {
  const encoder = new TextEncoder();
  const metaBytes = encoder.encode(JSON.stringify(metadata));
  const weightEntries = Object.entries(weights);

  let totalSize = 4 + 4 + 4 + metaBytes.length; // magic + version + json_len + json
  totalSize += 4; // num_params
  for (const [name, data] of weightEntries) {
    const nameBytes = encoder.encode(name);
    totalSize += 4 + nameBytes.length; // name_len + name
    totalSize += 4; // ndim
    totalSize += 4; // shape[0]
    // Align to 4 bytes for float data
    while (totalSize % 4 !== 0) totalSize++;
    totalSize += data.byteLength; // float data
  }

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let off = 0;

  // Magic: "SOUL"
  u8[off++] = 0x53; u8[off++] = 0x4f; u8[off++] = 0x55; u8[off++] = 0x4c;
  // Version: 3
  view.setUint32(off, 3, true); off += 4;
  // JSON len
  view.setUint32(off, metaBytes.length, true); off += 4;
  // JSON
  u8.set(metaBytes, off); off += metaBytes.length;
  // Num params
  view.setUint32(off, weightEntries.length, true); off += 4;

  for (const [name, data] of weightEntries) {
    const nameBytes = encoder.encode(name);
    view.setUint32(off, nameBytes.length, true); off += 4;
    u8.set(nameBytes, off); off += nameBytes.length;
    view.setUint32(off, 1, true); off += 4; // ndim
    view.setUint32(off, data.length, true); off += 4; // shape[0]
    // Align to 4 bytes before float data
    while (off % 4 !== 0) off++;
    u8.set(new Uint8Array(data.buffer), off); off += data.byteLength;
  }

  return buf;
}

beforeEach(() => {
  souLoader.unloadSou();
  AsyncStorage.clear();
  mockPickDocument = jest.fn();
});

describe('sou-loader', () => {
  it('isLoaded returns false initially', () => {
    expect(souLoader.isLoaded()).toBe(false);
  });

  it('unloadSou clears state', () => {
    souLoader.unloadSou();
    expect(souLoader.isLoaded()).toBe(false);
  });

  it('generate throws when no checkpoint loaded', async () => {
    await expect(souLoader.generateFromSou('hello')).rejects.toThrow('No .sou checkpoint loaded');
  });

  it('loadFromSouFile parses buffer and marks loaded', async () => {
    const emb = new Float32Array(256 * 4).fill(0.1);
    const fcW = new Float32Array(4 * 256).fill(0.05);
    const fcB = new Float32Array(4).fill(0.01);
    const buf = buildMinimalSou(
      { soul_name: 'test', soul_traits: {}, system_prompt: '', lineage: '' },
      { p0: emb, p1: fcW, p2: fcB },
    );
    await souLoader.loadFromSouFile(buf);
    expect(souLoader.isLoaded()).toBe(true);
  });

  it('loadFromSouFile caches in AsyncStorage', async () => {
    const emb = new Float32Array(256 * 4).fill(0.1);
    const fcW = new Float32Array(4 * 256).fill(0.05);
    const fcB = new Float32Array(4).fill(0.01);
    const buf = buildMinimalSou(
      { soul_name: 'cached-soul', soul_traits: {}, system_prompt: '', lineage: '' },
      { p0: emb, p1: fcW, p2: fcB },
    );
    await souLoader.loadFromSouFile(buf);
    const keys = await AsyncStorage.getAllKeys();
    expect(keys.some(k => k.includes('sou'))).toBe(true);
  });

  it('loadFromSouFile rejects invalid magic', async () => {
    const buf = new ArrayBuffer(16);
    new Uint8Array(buf).fill(0);
    await expect(souLoader.loadFromSouFile(buf)).rejects.toThrow('Invalid .sou magic');
  });

  it('loadFromSouFile restores from cache when available', async () => {
    const emb = new Float32Array(256 * 4).fill(0.2);
    const fcW = new Float32Array(4 * 256).fill(0.05);
    const fcB = new Float32Array(4).fill(0.01);
    const buf = buildMinimalSou(
      { soul_name: 'cache-test', soul_traits: {}, system_prompt: '', lineage: '' },
      { p0: emb, p1: fcW, p2: fcB },
    );
    // First load — writes to cache
    await souLoader.loadFromSouFile(buf);
    souLoader.unloadSou();
    // Second load — reads from cache (no buffer arg)
    await souLoader.loadFromSouFile();
    expect(souLoader.isLoaded()).toBe(true);
  });

  it('pickAndLoadSou returns null when picker canceled', async () => {
    mockPickDocument.mockResolvedValue({ canceled: true, assets: null });
    const result = await souLoader.pickAndLoadSou();
    expect(result).toBeNull();
    expect(souLoader.isLoaded()).toBe(false);
  });

  it('pickAndLoadSou loads valid .sou file', async () => {
    const emb = new Float32Array(256 * 4).fill(0.1);
    const fcW = new Float32Array(4 * 256).fill(0.05);
    const fcB = new Float32Array(4).fill(0.01);
    const buf = buildMinimalSou(
      { soul_name: 'picker-test', soul_traits: {}, system_prompt: '', lineage: '' },
      { p0: emb, p1: fcW, p2: fcB },
    );
    // Mock expo-file-system read
    jest.doMock('expo-file-system', () => ({
      readAsStringAsync: jest.fn().mockResolvedValue(
        btoa(String.fromCharCode(...new Uint8Array(buf))),
      ),
    }));
    mockPickDocument.mockResolvedValue({
      canceled: false,
      assets: [{ name: 'test.sou', uri: 'file:///tmp/test.sou', size: buf.byteLength }],
    });
    const result = await souLoader.pickAndLoadSou();
    expect(result).not.toBeNull();
    expect(souLoader.isLoaded()).toBe(true);
  });

  it('souMetadata returns null when not loaded', () => {
    expect(souLoader.souMetadata()).toBeNull();
  });

  it('souMetadata returns metadata after load', async () => {
    const emb = new Float32Array(256 * 4).fill(0.1);
    const fcW = new Float32Array(4 * 256).fill(0.05);
    const fcB = new Float32Array(4).fill(0.01);
    const buf = buildMinimalSou(
      { soul_name: 'meta-test', soul_traits: { friendly: 0.8 }, system_prompt: 'You are warm', lineage: 'v1' },
      { p0: emb, p1: fcW, p2: fcB },
    );
    await souLoader.loadFromSouFile(buf);
    const meta = souLoader.souMetadata();
    expect(meta).not.toBeNull();
    expect(meta!.soul_name).toBe('meta-test');
  });
});
