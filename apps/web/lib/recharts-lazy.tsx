'use client'

import { lazy } from 'react'

// Lazy-loaded recharts module — reduces initial bundle by ~400KB
// Usage: import { ResponsiveContainer, LineChart, ... } from '@/lib/recharts-lazy'
export const ResponsiveContainer = lazy(() => import('recharts').then(m => ({ default: m.ResponsiveContainer })))
export const LineChart = lazy(() => import('recharts').then(m => ({ default: m.LineChart })))
export const ComposedChart = lazy(() => import('recharts').then(m => ({ default: m.ComposedChart })))
export const BarChart = lazy(() => import('recharts').then(m => ({ default: m.BarChart })))
export const RadarChart = lazy(() => import('recharts').then(m => ({ default: m.RadarChart })))
export const Line = lazy(() => import('recharts').then(m => ({ default: m.Line })))
export const Bar = lazy(() => import('recharts').then(m => ({ default: m.Bar })))
export const Area = lazy(() => import('recharts').then(m => ({ default: m.Area })))
export const Radar = lazy(() => import('recharts').then(m => ({ default: m.Radar })))
export const XAxis = lazy(() => import('recharts').then(m => ({ default: m.XAxis })))
export const YAxis = lazy(() => import('recharts').then(m => ({ default: m.YAxis })))
export const Tooltip = lazy(() => import('recharts').then(m => ({ default: m.Tooltip })))
export const CartesianGrid = lazy(() => import('recharts').then(m => ({ default: m.CartesianGrid })))
export const Legend = lazy(() => import('recharts').then(m => ({ default: m.Legend })))
export const PolarGrid = lazy(() => import('recharts').then(m => ({ default: m.PolarGrid })))
export const PolarAngleAxis = lazy(() => import('recharts').then(m => ({ default: m.PolarAngleAxis })))
export const PolarRadiusAxis = lazy(() => import('recharts').then(m => ({ default: m.PolarRadiusAxis })))
