/**
 * Image upload service.
 * Captures images from camera or gallery and sends to the backend for analysis.
 */

import {Platform} from 'react-native';
import {api, getApiUrl} from './api-client';

export interface ImageResult {
  uri: string;
  base64: string;
  width: number;
  height: number;
}

export interface ImageAnalysis {
  description: string;
  tags: string[];
  caption: string;
}

/**
 * Pick an image from the gallery.
 */
export async function pickImage(): Promise<ImageResult | null> {
  let ImagePicker: any;
  try {
    ImagePicker = require('expo-image-picker');
  } catch {
    throw new Error('Image picker not available. Install expo-image-picker.');
  }

  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) {
    throw new Error('Photo library permission required');
  }

  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    quality: 0.8,
    base64: true,
    allowsEditing: true,
  });

  if (result.canceled || !result.assets?.[0]) return null;

  const asset = result.assets[0];
  return {
    uri: asset.uri,
    base64: asset.base64 || '',
    width: asset.width,
    height: asset.height,
  };
}

/**
 * Take a photo with the camera.
 */
export async function takePhoto(): Promise<ImageResult | null> {
  let ImagePicker: any;
  try {
    ImagePicker = require('expo-image-picker');
  } catch {
    throw new Error('Camera not available. Install expo-image-picker.');
  }

  const permission = await ImagePicker.requestCameraPermissionsAsync();
  if (!permission.granted) {
    throw new Error('Camera permission required');
  }

  const result = await ImagePicker.launchCameraAsync({
    quality: 0.8,
    base64: true,
    allowsEditing: true,
  });

  if (result.canceled || !result.assets?.[0]) return null;

  const asset = result.assets[0];
  return {
    uri: asset.uri,
    base64: asset.base64 || '',
    width: asset.width,
    height: asset.height,
  };
}

/**
 * Send an image to the backend for analysis.
 */
export async function analyzeImage(image: ImageResult): Promise<ImageAnalysis> {
  const baseUrl = await getApiUrl();
  const formData = new FormData();
  formData.append('image', {
    uri: image.uri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  } as any);

  try {
    const res = await fetch(`${baseUrl}/multimodal/analyze-image`, {
      method: 'POST',
      body: formData,
      headers: {'Content-Type': 'multipart/form-data'},
    });

    if (res.ok) {
      const data = await res.json();
      return {
        description: data.description || '',
        tags: data.tags || [],
        caption: data.caption || '',
      };
    }
  } catch {
    // backend unavailable
  }

  return {description: '', tags: [], caption: ''};
}

/**
 * Convert image to base64 data URL for chat messages.
 */
export function imageDataUrl(image: ImageResult): string {
  return `data:image/jpeg;base64,${image.base64}`;
}
