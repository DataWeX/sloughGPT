---
id: 20261112_140458_losschart-rewrite-rendering-internals-with-d3-d3-scaled3-sha
status: review
board: 20261112_140458_losschart-rewrite-rendering-internals-with-d3-d3-scaled3-sha
title: "LossChart: rewrite rendering internals with D3 (d3-scale/d3-shape)"
tags: ["frontend", "training", "chart", "area:web"]
sprint: ""
gh: null
---

# LossChart: rewrite rendering internals with D3 (d3-scale/d3-shape)

Replace recharts rendering inside apps/web/components/training/LossChart.tsx with d3-scale + d3-shape: linear x/y scales, explicit even ticks (fixes irregular category-axis ticks seen in screenshot), d3 line() path generator, React-rendered SVG. Public API (LossChartProps), CSV export, live sliding window, dual loss/reward axes, legend stay byte-identical. Consumers TrainingPipeline.tsx + JobLossCard.tsx unchanged. Keep LossChart.test.tsx green; add regression test asserting evenly spaced tick labels. Branch: feat/loss-chart-d3. Scope: LossChart only, no other charts migrate.
