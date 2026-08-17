export const getDocumentAsync = jest.fn().mockResolvedValue({canceled: true, assets: []});
export const MediaTypeOptions = {
  All: '*/*',
  Images: 'image/*',
  Videos: 'video/*',
  Audio: 'audio/*',
  PDF: 'application/pdf',
  PlainText: 'text/plain',
};
