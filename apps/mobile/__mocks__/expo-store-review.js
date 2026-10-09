module.exports = {
  isAvailableAsync: jest.fn().mockResolvedValue(true),
  requestReviewAsync: jest.fn().mockResolvedValue(undefined),
};
