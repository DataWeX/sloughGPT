module.exports = {
  setNotificationHandler: jest.fn(),
  getPermissionsAsync: jest.fn(async () => ({status: 'granted'})),
  requestPermissionsAsync: jest.fn(async () => ({status: 'granted'})),
  getExpoPushTokenAsync: jest.fn(async () => ({data: 'ExpoPushToken[test123]'})),
  setBadgeCountAsync: jest.fn(async () => {}),
  getBadgeCountAsync: jest.fn(async () => 0),
  addNotificationReceivedListener: jest.fn(() => ({remove: jest.fn()})),
  addNotificationResponseReceivedListener: jest.fn(() => ({remove: jest.fn()})),
};
