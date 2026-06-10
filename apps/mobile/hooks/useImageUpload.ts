import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Platform } from 'react-native';
import { Analytics, PerformanceTracker } from '../lib/analytics';

export interface UseImageUploadReturn {
  image: string | null;
  isUploading: boolean;
  pickImage: () => Promise<void>;
  takePhoto: () => Promise<void>;
  clearImage: () => void;
  error: string | null;
}

export function useImageUpload(): UseImageUploadReturn {
  const [image, setImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestPermissions = async (type: 'camera' | 'library') => {
    if (Platform.OS === 'web') return true;

    if (type === 'camera') {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        setError('Camera permission denied');
        Analytics.trackEvent('camera_permission_denied');
        return false;
      }
    } else {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        setError('Photo library permission denied');
        Analytics.trackEvent('library_permission_denied');
        return false;
      }
    }
    return true;
  };

  const pickImage = async () => {
    try {
      setError(null);
      const hasPermission = await requestPermissions('library');
      if (!hasPermission) return;

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setImage(result.assets[0].uri);
        Analytics.trackEvent('image_selected_from_library', {
          width: result.assets[0].width,
          height: result.assets[0].height,
        });
      }
    } catch (err) {
      setError('Failed to pick image');
      PerformanceTracker.trackError(err as Error, { context: 'image_pick' });
    }
  };

  const takePhoto = async () => {
    try {
      setError(null);
      const hasPermission = await requestPermissions('camera');
      if (!hasPermission) return;

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setImage(result.assets[0].uri);
        Analytics.trackEvent('photo_taken', {
          width: result.assets[0].width,
          height: result.assets[0].height,
        });
      }
    } catch (err) {
      setError('Failed to take photo');
      PerformanceTracker.trackError(err as Error, { context: 'photo_capture' });
    }
  };

  const clearImage = () => {
    setImage(null);
    setError(null);
    Analytics.trackEvent('image_cleared');
  };

  return {
    image,
    isUploading,
    pickImage,
    takePhoto,
    clearImage,
    error,
  };
}
