jest.mock('../haptics');

import {playSound, setSoundsEnabled, areSoundsEnabled, sounds} from '../sounds';

describe('sounds', () => {
  beforeEach(() => {
    setSoundsEnabled(true);
  });

  it('send() does not throw', async () => {
    await expect(sounds.send()).resolves.toBeUndefined();
  });

  it('receive() does not throw', async () => {
    await expect(sounds.receive()).resolves.toBeUndefined();
  });

  it('error() does not throw', async () => {
    await expect(sounds.error()).resolves.toBeUndefined();
  });

  it('delete() does not throw', async () => {
    await expect(sounds.delete()).resolves.toBeUndefined();
  });

  it('areSoundsEnabled returns true by default', () => {
    expect(areSoundsEnabled()).toBe(true);
  });

  it('setSoundsEnabled(false) disables sounds', async () => {
    setSoundsEnabled(false);
    expect(areSoundsEnabled()).toBe(false);
    // Silent when disabled
    await expect(sounds.send()).resolves.toBeUndefined();
  });

  it('setEnabled is an alias for setSoundsEnabled', () => {
    sounds.setEnabled(false);
    expect(areSoundsEnabled()).toBe(false);
  });

  it('isEnabled is an alias for areSoundsEnabled', () => {
    expect(sounds.isEnabled()).toBe(true);
  });
});
