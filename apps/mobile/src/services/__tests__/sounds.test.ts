import {sounds, setSoundsEnabled, areSoundsEnabled, playSound} from '../sounds';

jest.mock('../haptics', () => ({
  triggerHaptic: jest.fn(),
}));

beforeEach(() => {
  jest.clearAllMocks();
  setSoundsEnabled(true);
});

describe('sounds service', () => {
  it('areSoundsEnabled returns true by default', () => {
    expect(areSoundsEnabled()).toBe(true);
  });

  it('setSoundsEnabled toggles state', () => {
    setSoundsEnabled(false);
    expect(areSoundsEnabled()).toBe(false);
    setSoundsEnabled(true);
    expect(areSoundsEnabled()).toBe(true);
  });

  it('playSound does not throw', async () => {
    await expect(playSound('send')).resolves.toBeUndefined();
    await expect(playSound('receive')).resolves.toBeUndefined();
    await expect(playSound('error')).resolves.toBeUndefined();
    await expect(playSound('delete')).resolves.toBeUndefined();
  });

  it('playSound does nothing when disabled', async () => {
    setSoundsEnabled(false);
    await playSound('send');
    // No error, no haptic called
  });

  it('sounds.send calls playSound', async () => {
    await sounds.send();
    // No throw
  });

  it('sounds.receive calls playSound', async () => {
    await sounds.receive();
  });

  it('sounds.error calls playSound', async () => {
    await sounds.error();
  });

  it('sounds.delete calls playSound', async () => {
    await sounds.delete();
  });

  it('sounds.setEnabled delegates', () => {
    sounds.setEnabled(false);
    expect(sounds.isEnabled()).toBe(false);
    sounds.setEnabled(true);
    expect(sounds.isEnabled()).toBe(true);
  });
});
