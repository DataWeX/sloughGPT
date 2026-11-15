---
id: 20261106_101712_feat-training-live-job-links-click-feed-chipsherokpi-into-tr
status: review
board: 20261106_101712_feat-training-live-job-links-click-feed-chipsherokpi-into-tr
title: "feat: training live-job links — click feed chips/hero/KPI into /training/job/<id> in-place view"
tags: ["web", "training", "ux", "area:web"]
sprint: ""
gh: null
---

# feat: training live-job links — click feed chips/hero/KPI into /training/job/<id> in-place view

LiveFeedStream active-job chips are inert badges; the only live in-place job view (/training/job/<id>, 3s poll + loss chart) is reachable only via notification process_link or /monitoring TrainingHistory. Wire chips + hero + Active-jobs KPI to the job detail page (Link from vite/next-compat). Journey assertion in LiveFeedStream.test.tsx. No new page (surface ladder: compose, don't add routes).
