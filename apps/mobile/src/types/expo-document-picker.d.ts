declare module 'expo-document-picker' {
  export interface DocumentPickerAsset {
    name: string;
    size: number;
    uri: string;
    mimeType: string;
    lastModified: number;
  }

  export interface DocumentPickerResult {
    canceled: boolean;
    assets: DocumentPickerAsset[] | null;
  }

  export interface DocumentPickerOptions {
    type?: string | string[];
    copyToCacheDirectory?: boolean;
    multiple?: boolean;
  }

  export function getDocumentAsync(options?: DocumentPickerOptions): Promise<DocumentPickerResult>;
}
