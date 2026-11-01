/**
 * Image upload service.
 * Captures images from camera or gallery and sends to the backend for analysis.
 */

import {Platform, Alert, Linking} from 'react-native';
import {api, getApiUrl} from './api-client';

let _ImagePicker: any = null;
function getImagePicker(): any {
  if (_ImagePicker) return _ImagePicker;
  try {
    _ImagePicker = require('expo-image-picker');
    return _ImagePicker;
  } catch {
    return null;
  }
}

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

function handlePermissionDenied(permission: string): null {
  Alert.alert(
    `${permission} permission required`,
    `Please enable ${permission.toLowerCase()} access in your device settings to use this feature.`,
    [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Open Settings',
        onPress: () => Linking.openSettings().catch(() => {}),
      },
    ],
  );
  return null;
}

/**
 * Pick an image from the gallery.
 */
export async function pickImage(): Promise<ImageResult | null> {
  const ImagePicker = getImagePicker();
  if (!ImagePicker) {
    Alert.alert('Image picker unavailable', 'expo-image-picker is not installed.');
    return null;
  }

  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) {
    return handlePermissionDenied('Photo library');
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
  const ImagePicker = getImagePicker();
  if (!ImagePicker) {
    Alert.alert('Camera unavailable', 'expo-image-picker is not installed.');
    return null;
  }

  const permission = await ImagePicker.requestCameraPermissionsAsync();
  if (!permission.granted) {
    return handlePermissionDenied('Camera');
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
