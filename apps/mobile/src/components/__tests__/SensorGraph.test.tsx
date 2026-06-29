import React from 'react';
import {render} from '@testing-library/react-native';
import {SensorGraph} from '../SensorGraph';
import type {SensorReading} from '../../types';

const makeData = (n: number): SensorReading[] =>
  Array.from({length: n}, (_, i) => ({
    timestamp: Date.now() + i * 100,
    accel: {
      x: Math.sin(i * 0.1),
      y: Math.cos(i * 0.1),
      z: 9.8 + Math.sin(i * 0.05) * 0.1,
    },
    gyro: {
      x: Math.sin(i * 0.2) * 0.5,
      y: Math.cos(i * 0.2) * 0.3,
      z: Math.sin(i * 0.15) * 0.2,
    },
  }));

describe('SensorGraph', () => {
  it('renders placeholder when data has fewer than 2 points', () => {
    expect(() => {
      render(
        <SensorGraph
          data={makeData(1)}
          channels={['accel_x', 'accel_y', 'accel_z']}
          width={280}
          height={100}
        />,
      );
    }).not.toThrow();
  });

  it('renders graph with data', () => {
    expect(() => {
      render(
        <SensorGraph
          data={makeData(20)}
          channels={['accel_x', 'accel_y', 'accel_z']}
          width={280}
          height={100}
        />,
      );
    }).not.toThrow();
  });

  it('renders with all 6 channels', () => {
    expect(() => {
      render(
        <SensorGraph
          data={makeData(50)}
          channels={['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}
          width={300}
          height={120}
        />,
      );
    }).not.toThrow();
  });

  it('renders with custom dimensions', () => {
    expect(() => {
      render(
        <SensorGraph
          data={makeData(30)}
          channels={['accel_x']}
          width={200}
          height={80}
        />,
      );
    }).not.toThrow();
  });

  it('renders with many data points', () => {
    expect(() => {
      render(
        <SensorGraph
          data={makeData(200)}
          channels={['accel_x', 'gyro_x']}
          width={350}
          height={150}
        />,
      );
    }).not.toThrow();
  });
});
