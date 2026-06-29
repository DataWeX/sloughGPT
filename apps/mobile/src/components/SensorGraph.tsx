import React from 'react';
import {View, StyleSheet} from 'react-native';
import {colors} from '../theme';
import type {SensorReading} from '../types';

interface Props {
  data: SensorReading[];
  channels: ('accel_x' | 'accel_y' | 'accel_z' | 'gyro_x' | 'gyro_y' | 'gyro_z')[];
  width?: number;
  height?: number;
}

const CHANNEL_COLORS: Record<string, string> = {
  accel_x: '#DC505A',
  accel_y: '#34B07D',
  accel_z: '#7C52C4',
  gyro_x: '#ECA83C',
  gyro_y: '#EC915F',
  gyro_z: '#5B9BD5',
};

function getValue(r: SensorReading, ch: string): number {
  switch (ch) {
    case 'accel_x': return r.accel.x;
    case 'accel_y': return r.accel.y;
    case 'accel_z': return r.accel.z;
    case 'gyro_x': return r.gyro.x;
    case 'gyro_y': return r.gyro.y;
    case 'gyro_z': return r.gyro.z;
    default: return 0;
  }
}

export function SensorGraph({data, channels, width = 280, height = 100}: Props) {
  if (data.length < 2) return <View style={[styles.placeholder, {width, height}]} />;

  const pad = 4;
  const graphW = width - pad * 2;
  const graphH = height - pad * 2;

  // Compute per-channel range
  const ranges = channels.map(ch => {
    const values = data.map(r => getValue(r, ch));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return {min, max, range};
  });

  return (
    <View style={[styles.graph, {width, height}]}>
      {channels.map((ch, ci) => {
        const color = CHANNEL_COLORS[ch] || colors.primary;
        const {min, range} = ranges[ci];
        const points: string[] = [];
        const step = Math.max(1, Math.floor((data.length - 1) / (graphW / 2)));
        for (let i = 0; i < data.length; i += step) {
          const x = pad + (i / (data.length - 1)) * graphW;
          const y = pad + (1 - (getValue(data[i], ch) - min) / range) * graphH;
          points.push(`${x},${y}`);
        }
        // Also add last point
        const last = data.length - 1;
        const xLast = pad + graphW;
        const yLast = pad + (1 - (getValue(data[last], ch) - min) / range) * graphH;
        points.push(`${xLast},${yLast}`);

        const polyline = points.join(' ');
        return (
          <React.Fragment key={ch}>
            {/* Inline SVG-like polyline via absolute-positioned thin views */}
            {points.slice(1).map((_, i) => {
              const [x1s, y1s] = points[i].split(',');
              const [x2s, y2s] = points[i + 1].split(',');
              const x1 = parseFloat(x1s);
              const y1 = parseFloat(y1s);
              const x2 = parseFloat(x2s);
              const y2 = parseFloat(y2s);
              const dx = x2 - x1;
              const dy = y2 - y1;
              const len = Math.sqrt(dx * dx + dy * dy);
              const angle = Math.atan2(dy, dx) * (180 / Math.PI);
              return (
                <View
                  key={`${ch}-${i}`}
                  style={{
                    position: 'absolute',
                    left: x1,
                    top: y1,
                    width: len,
                    height: 1.5,
                    backgroundColor: color,
                    opacity: 0.8,
                    transform: [{rotate: `${angle}deg`}],
                    transformOrigin: 'left center',
                  }}
                />
              );
            })}
          </React.Fragment>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  graph: {
    backgroundColor: colors.surface,
    borderRadius: 6,
    overflow: 'hidden',
  },
  placeholder: {
    backgroundColor: colors.surface,
    borderRadius: 6,
  },
});
