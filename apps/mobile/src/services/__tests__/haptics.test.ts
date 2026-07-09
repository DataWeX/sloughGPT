import {triggerHaptic} from '../haptics';

// Mock expo-haptics to be unavailable (no native module during tests)
jest.mock('expo-haptics', () => {
  throw new Error('not installed');
}, {virtual: true});

describe('triggerHaptic', () => {
  it('light type does not throw', async () => {
    await expect(triggerHaptic('light')).resolves.toBeUndefined();
  });

  it('medium type does not throw', async () => {
    await expect(triggerHaptic('medium')).resolves.toBeUndefined();
  });

  it('heavy type does not throw', async () => {
    await expect(triggerHaptic('heavy')).resolves.toBeUndefined();
  });

  it('success type does not throw', async () => {
    await expect(triggerHaptic('success')).resolves.toBeUndefined();
  });

  it('error type does not throw', async () => {
    await expect(triggerHaptic('error')).resolves.toBeUndefined();
  });

  it('selection type does not throw', async () => {
    await expect(triggerHaptic('selection')).resolves.toBeUndefined();
  });

  it('defaults to light', async () => {
    await expect(triggerHaptic()).resolves.toBeUndefined();
  });
});
