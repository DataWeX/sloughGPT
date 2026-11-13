jest.mock('react-native', () => ({
  Clipboard: {setString: jest.fn()},
  Alert: {alert: jest.fn()},
  Linking: {openSettings: jest.fn().mockResolvedValue(undefined)},
  Platform: {OS: 'ios'},
}));

jest.mock('../haptics', () => ({triggerHaptic: jest.fn()}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {Clipboard, Alert, Linking} = require('react-native');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const {triggerHaptic} = require('../haptics');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const {copyToClipboard, copyWithFeedback, permissionDeniedAlert} = require('../clipboard');

beforeEach(() => {
  jest.clearAllMocks();
});

describe('clipboard', () => {
  describe('copyToClipboard', () => {
    it('copies text to clipboard', async () => {
      const result = await copyToClipboard('Hello');
      expect(Clipboard.setString).toHaveBeenCalledWith('Hello');
      expect(result).toBe(true);
    });

    it('triggers haptic on success', async () => {
      await copyToClipboard('Hello');
      expect(triggerHaptic).toHaveBeenCalledWith('success');
    });

    it('returns false on error', async () => {
      (Clipboard.setString as jest.Mock).mockImplementation(() => {
        throw new Error('copy failed');
      });
      const result = await copyToClipboard('Hello');
      expect(result).toBe(false);
    });
  });

  describe('copyWithFeedback', () => {
    it('copies text to clipboard', async () => {
      await copyWithFeedback('Hello');
      expect(Clipboard.setString).toHaveBeenCalledWith('Hello');
    });
  });

  describe('permissionDeniedAlert', () => {
    it('shows alert with correct message', () => {
      permissionDeniedAlert('Camera');
      expect(Alert.alert).toHaveBeenCalledWith(
        'Camera permission required',
        expect.stringContaining('camera'),
        expect.any(Array),
      );
    });

    it('has Open Settings button', () => {
      permissionDeniedAlert('Photos');
      const buttons = (Alert.alert as jest.Mock).mock.calls[0][2];
      expect(buttons).toHaveLength(2);
      expect(buttons[1].text).toBe('Open Settings');
    });

    it('opens settings when pressed', () => {
      permissionDeniedAlert('Camera');
      const buttons = (Alert.alert as jest.Mock).mock.calls[0][2];
      buttons[1].onPress();
      expect(Linking.openSettings).toHaveBeenCalled();
    });
  });
});
