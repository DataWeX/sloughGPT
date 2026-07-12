import React from 'react';
import {
  Copy, ExternalLink, Reply, Forward, Pencil,
  ArrowUp, ArrowDown,
  Pin, PinOff, Star, ThumbsUp, ThumbsDown,
  RefreshCw, Trash2, Check, Mic,
  StopCircle, Zap, MessageCircle,
  Brain, Dumbbell, BookOpen,
  Search, X, Menu, Settings,
  Sun, Moon, Package, Archive,
  TriangleAlert, Palette, BarChart3,   Plus,
  Bookmark, Share2, Image, Upload,
  Download, Lock, Unlock, Info,
  MoreHorizontal, MoreVertical,
  Music, Headphones, Target, Save,
  HelpCircle, User, Book, Square,
  ArchiveX, MessageSquare,
  SmilePlus,
} from 'lucide-react-native';
import type {LucideIcon} from 'lucide-react-native';
export type IconName =
  | 'copy' | 'external-link' | 'reply' | 'forward' | 'edit'
  | 'pin' | 'pin-off' | 'star' | 'star-outline'
  | 'smile-plus' | 'thumbs-up' | 'thumbs-down'
  | 'refresh-cw' | 'trash-2' | 'check' | 'mic'
  | 'arrow-up' | 'arrow-down'
  | 'stop-circle' | 'zap' | 'message-circle'
  | 'brain' | 'dumbbell' | 'book-open'
  | 'search' | 'x' | 'menu' | 'settings'
  | 'sun' | 'moon' | 'package' | 'archive'
  | 'triangle-alert' | 'palette' | 'bar-chart' | 'plus'
  | 'bookmark' | 'share-2' | 'image' | 'upload'
  | 'download' | 'lock' | 'unlock' | 'info'
  | 'more-horizontal' | 'more-vertical'
  | 'music' | 'headphones' | 'target' | 'save'
  | 'help' | 'user' | 'book' | 'square'
  | 'archive-x' | 'message-square'
  ;

interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
}

const ICON_MAP: Record<string, LucideIcon> = {
  'copy': Copy,
  'external-link': ExternalLink,
  'reply': Reply,
  'forward': Forward,
  'edit': Pencil,
  'pin': Pin,
  'pin-off': PinOff,
  'star': Star,
  'star-outline': Star,
  'smile-plus': SmilePlus,
  'thumbs-up': ThumbsUp,
  'thumbs-down': ThumbsDown,
  'refresh-cw': RefreshCw,
  'trash-2': Trash2,
  'check': Check,
  'mic': Mic,
  'stop-circle': StopCircle,
  'zap': Zap,
  'message-circle': MessageCircle,
  'brain': Brain,
  'dumbbell': Dumbbell,
  'book-open': BookOpen,
  'search': Search,
  'x': X,
  'menu': Menu,
  'settings': Settings,
  'sun': Sun,
  'moon': Moon,
  'package': Package,
  'archive': Archive,
  'triangle-alert': TriangleAlert,
  'palette': Palette,
  'bar-chart': BarChart3,
  'plus': Plus,
  'arrow-up': ArrowUp,
  'arrow-down': ArrowDown,
  'bookmark': Bookmark,
  'share-2': Share2,
  'image': Image,
  'upload': Upload,
  'download': Download,
  'lock': Lock,
  'unlock': Unlock,
  'info': Info,
  'more-horizontal': MoreHorizontal,
  'more-vertical': MoreVertical,
  'music': Music,
  'headphones': Headphones,
  'target': Target,
  'save': Save,
  'help': HelpCircle,
  'user': User,
  'book': Book,
  'square': Square,
  'archive-x': ArchiveX,
  'message-square': MessageSquare,
};

export function Icon({name, size = 20, color = '$color'}: IconProps) {
  const C = ICON_MAP[name];
  if (!C) return null;
  return <C size={size} color={color} />;
}
