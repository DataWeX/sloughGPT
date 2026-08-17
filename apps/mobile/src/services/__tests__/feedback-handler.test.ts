jest.mock('../haptics', () => ({
  triggerHaptic: jest.fn(() => Promise.resolve()),
}));

jest.mock('../sounds', () => ({
  sounds: {
    send: jest.fn(() => Promise.resolve()),
    receive: jest.fn(() => Promise.resolve()),
    error: jest.fn(() => Promise.resolve()),
    delete: jest.fn(() => Promise.resolve()),
    setEnabled: jest.fn(),
    isEnabled: jest.fn(() => true),
  },
  setSoundsEnabled: jest.fn(),
  areSoundsEnabled: jest.fn(() => true),
}));

const {triggerHaptic} = require('../haptics');
const {sounds} = require('../sounds');
const {feedback} = require('../feedback-handler');

beforeEach(() => {
  jest.clearAllMocks();
  feedback.setEnabled(true);
});

describe('feedback-handler', () => {
  describe('setEnabled / isEnabled', () => {
    it('is enabled by default', () => {
      expect(feedback.isEnabled()).toBe(true);
    });

    it('disables all feedback', () => {
      feedback.setEnabled(false);
      expect(feedback.isEnabled()).toBe(false);
    });

    it('re-enables feedback', () => {
      feedback.setEnabled(false);
      feedback.setEnabled(true);
      expect(feedback.isEnabled()).toBe(true);
    });

    it('syncs with sounds.setEnabled', () => {
      feedback.setEnabled(false);
      expect(sounds.setEnabled).toHaveBeenCalledWith(false);
    });
  });

  describe('fire', () => {
    it('calls triggerHaptic for tap event', async () => {
      await feedback.fire('tap');
      expect(triggerHaptic).toHaveBeenCalledWith('light');
    });

    it('calls triggerHaptic + sounds.send for send event', async () => {
      await feedback.fire('send');
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
      expect(sounds.send).toHaveBeenCalled();
    });

    it('calls triggerHaptic + sounds.receive for receive event', async () => {
      await feedback.fire('receive');
      expect(triggerHaptic).toHaveBeenCalledWith('light');
      expect(sounds.receive).toHaveBeenCalled();
    });

    it('calls triggerHaptic + sounds.receive for success event', async () => {
      await feedback.fire('success');
      expect(triggerHaptic).toHaveBeenCalledWith('success');
      expect(sounds.receive).toHaveBeenCalled();
    });

    it('calls triggerHaptic + sounds.error for error event', async () => {
      await feedback.fire('error');
      expect(triggerHaptic).toHaveBeenCalledWith('error');
      expect(sounds.error).toHaveBeenCalled();
    });

    it('calls triggerHaptic + sounds.delete for delete event', async () => {
      await feedback.fire('delete');
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
      expect(sounds.delete).toHaveBeenCalled();
    });

    it('calls triggerHaptic for warning event (no sound)', async () => {
      await feedback.fire('warning');
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
      expect(sounds.send).not.toHaveBeenCalled();
      expect(sounds.receive).not.toHaveBeenCalled();
    });

    it('calls triggerHaptic for select event (no sound)', async () => {
      await feedback.fire('select');
      expect(triggerHaptic).toHaveBeenCalledWith('selection');
    });

    it('calls triggerHaptic for confirm event (no sound)', async () => {
      await feedback.fire('confirm');
      expect(triggerHaptic).toHaveBeenCalledWith('heavy');
    });

    it('does nothing when disabled', async () => {
      feedback.setEnabled(false);
      await feedback.fire('send');
      expect(triggerHaptic).not.toHaveBeenCalled();
      expect(sounds.send).not.toHaveBeenCalled();
    });

    it('ignores unknown event names', async () => {
      await feedback.fire('nonexistent' as any);
      expect(triggerHaptic).not.toHaveBeenCalled();
    });
  });

  describe('convenience methods', () => {
    it('tap() fires tap event', async () => {
      await feedback.tap();
      expect(triggerHaptic).toHaveBeenCalledWith('light');
    });

    it('send() fires send event', async () => {
      await feedback.send();
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
      expect(sounds.send).toHaveBeenCalled();
    });

    it('receive() fires receive event', async () => {
      await feedback.receive();
      expect(triggerHaptic).toHaveBeenCalledWith('light');
      expect(sounds.receive).toHaveBeenCalled();
    });

    it('success() fires success event', async () => {
      await feedback.success();
      expect(triggerHaptic).toHaveBeenCalledWith('success');
    });

    it('error() fires error event', async () => {
      await feedback.error();
      expect(triggerHaptic).toHaveBeenCalledWith('error');
      expect(sounds.error).toHaveBeenCalled();
    });

    it('delete() fires delete event', async () => {
      await feedback.delete();
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
      expect(sounds.delete).toHaveBeenCalled();
    });

    it('warning() fires warning event', async () => {
      await feedback.warning();
      expect(triggerHaptic).toHaveBeenCalledWith('medium');
    });

    it('select() fires select event', async () => {
      await feedback.select();
      expect(triggerHaptic).toHaveBeenCalledWith('selection');
    });

    it('confirm() fires confirm event', async () => {
      await feedback.confirm();
      expect(triggerHaptic).toHaveBeenCalledWith('heavy');
    });
  });
});
