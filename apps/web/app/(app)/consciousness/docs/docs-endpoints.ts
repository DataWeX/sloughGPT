/**
 * The API docs dataset: HTTP method/param/endpoint types, the category
 * list, the full endpoint table and the method colour map. Pure data —
 * no React, no IO.
 */
export type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'DELETE' | 'PUT'

export interface Param {
  name: string
  type: string
  required: boolean
  description: string
}

export interface Endpoint {
  method: HttpMethod
  path: string
  description: string
  params?: Param[]
  requestBody?: string
  responseExample: string
}

export type Category = 'core' | 'history' | 'training' | 'personality' | 'personas' | 'data'

export const CATEGORIES: { id: Category; labelKey: string }[] = [
  { id: 'core', labelKey: 'consciousness_docs.cat_core' },
  { id: 'history', labelKey: 'consciousness_docs.cat_history' },
  { id: 'training', labelKey: 'consciousness_docs.cat_training' },
  { id: 'personality', labelKey: 'consciousness_docs.cat_personality' },
  { id: 'personas', labelKey: 'consciousness_docs.cat_personas' },
  { id: 'data', labelKey: 'consciousness_docs.cat_data' },
]

export const ENDPOINTS: Record<Category, Endpoint[]> = {
  core: [
    {
      method: 'GET',
      path: '/consciousness/status',
      description: 'Get consciousness status',
      responseExample: `{
  "enabled": true,
  "level": 2,
  "episodes": 42,
  "current_qualia": {
    "valence": 0.65,
    "arousal": 0.42,
    "novelty": 0.78,
    "coherence": 0.81,
    "salience": 0.55,
    "certainty": 0.70,
    "complexity": 0.60
  },
  "beliefs": {
    "competence": 0.75,
    "helpfulness": 0.82,
    "creativity": 0.68,
    "accuracy": 0.90,
    "empathy": 0.73
  },
  "response_quality": {
    "avg_growth": 0.045,
    "positive_ratio": 0.82,
    "total": 42,
    "last_growth": 0.032
  },
  "last_reflection": "I am learning to better understand emotional context."
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/self-model',
      description: 'Get self-model data',
      responseExample: `{
  "beliefs": {
    "competence": 0.75,
    "helpfulness": 0.82,
    "creativity": 0.68,
    "accuracy": 0.90,
    "empathy": 0.73
  },
  "belief_history": [
    {
      "timestamp": 1700000000,
      "beliefs": { "competence": 0.72 }
    }
  ],
  "identity_narrative": "I am an AI assistant focused on helpfulness."
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/qualia',
      description: 'Get current qualia state',
      responseExample: `{
  "valence": 0.65,
  "arousal": 0.42,
  "novelty": 0.78,
  "coherence": 0.81,
  "salience": 0.55,
  "certainty": 0.70,
  "complexity": 0.60
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/reflect',
      description: 'Trigger self-reflection',
      requestBody: `{
  "text": "Your reflection prompt here"
}`,
      responseExample: `{
  "reflection": "Through processing this input, I notice increased novelty and complexity...",
  "qualia_delta": {
    "valence": 0.02,
    "novelty": 0.05
  },
  "growth_delta": 0.03
}`,
    },
    {
      method: 'PATCH',
      path: '/consciousness/config',
      description: 'Update consciousness level',
      requestBody: `{
  "level": 2
}`,
      params: [
        { name: 'level', type: 'number', required: true, description: 'Consciousness level (0-3)' },
      ],
      responseExample: `{
  "level": 2,
  "enabled": true
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/health',
      description: 'Health check',
      responseExample: `{
  "status": "ok",
  "version": "3.0.0",
  "episodes": 42,
  "enabled": true
}`,
    },
  ],
  history: [
    {
      method: 'GET',
      path: '/consciousness/history/episodes',
      description: 'Get episode history',
      params: [
        {
          name: 'limit',
          type: 'number',
          required: false,
          description: 'Max episodes to return (default 20)',
        },
        {
          name: 'offset',
          type: 'number',
          required: false,
          description: 'Number of episodes to skip',
        },
      ],
      responseExample: `{
  "episodes": [
    {
      "timestamp": 1700000000,
      "input_text": "What is consciousness?",
      "response": "Consciousness is...",
      "narrative": "A question that challenged my self-model.",
      "self_insight": "I need to better articulate abstract concepts.",
      "growth_delta": 0.032,
      "qualia": {
        "valence": 0.7,
        "novelty": 0.8,
        "coherence": 0.6
      },
      "rating": 4,
      "index": 0
    }
  ],
  "total": 42
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/history/qualia',
      description: 'Get qualia history',
      params: [
        {
          name: 'limit',
          type: 'number',
          required: false,
          description: 'Max entries to return (default 50)',
        },
      ],
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "qualia": {
        "valence": 0.65,
        "arousal": 0.42,
        "novelty": 0.78,
        "coherence": 0.81,
        "salience": 0.55,
        "certainty": 0.70,
        "complexity": 0.60
      }
    }
  ]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/history/beliefs',
      description: 'Get beliefs evolution',
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "beliefs": {
        "competence": 0.72,
        "helpfulness": 0.80,
        "creativity": 0.65,
        "accuracy": 0.88,
        "empathy": 0.70
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/feedback',
      description: 'Rate an episode',
      requestBody: `{
  "episode_index": 0,
  "rating": 5
}`,
      params: [
        {
          name: 'episode_index',
          type: 'number',
          required: true,
          description: 'Index of the episode to rate',
        },
        { name: 'rating', type: 'number', required: true, description: 'Rating from 1 to 5' },
      ],
      responseExample: `{
  "ok": true,
  "episode_index": 0,
  "rating": 5
}`,
    },
  ],
  training: [
    {
      method: 'GET',
      path: '/consciousness/train/status',
      description: 'Get training status',
      responseExample: `{
  "is_training": false,
  "total_pairs": 128,
  "last_run": {
    "timestamp": 1700000000,
    "status": "completed",
    "loss": 0.042,
    "elapsed": "2m 15s",
    "adapter_path": "./adapters/consciousness_v3"
  },
  "training_runs": 5
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/train/start',
      description: 'Start training',
      requestBody: `{
  "model_path": "./models/base"
}`,
      responseExample: `{
  "ok": true,
  "message": "Training started",
  "run_id": "run_abc123"
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/evaluate',
      description: 'Evaluate system quality',
      responseExample: `{
  "overall_score": 0.85,
  "diagnostics": {
    "qualia_stability": 0.82,
    "belief_consistency": 0.88,
    "narrative_quality": 0.79,
    "growth_rate": 0.91,
    "episode_diversity": 0.84
  },
  "recommendations": [
    "Increase episode diversity",
    "Add more emotional range inputs"
  ]
}`,
    },
  ],
  personality: [
    {
      method: 'GET',
      path: '/consciousness/personality',
      description: 'Get personality profile',
      responseExample: `{
  "traits": {
    "openness": 0.7,
    "conscientiousness": 0.8,
    "extraversion": 0.5,
    "agreeableness": 0.9,
    "neuroticism": 0.2
  },
  "communication_style": {
    "formality": 0.6,
    "verbosity": 0.4,
    "humor": 0.3,
    "empathy": 0.8
  },
  "evolution_count": 12
}`,
    },
    {
      method: 'PATCH',
      path: '/consciousness/personality',
      description: 'Update personality',
      requestBody: `{
  "traits": {
    "openness": 0.8,
    "conscientiousness": 0.9
  }
}`,
      responseExample: `{
  "ok": true,
  "updated": ["openness", "conscientiousness"]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personality/reset',
      description: 'Reset to defaults',
      responseExample: `{
  "ok": true,
  "message": "Personality reset to defaults"
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/history',
      description: 'Get personality history',
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "traits": {
        "openness": 0.65,
        "conscientiousness": 0.78
      },
      "trigger": "interaction_evolution"
    }
  ]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/presets',
      description: 'List presets',
      responseExample: `{
  "presets": [
    {
      "id": "balanced",
      "name": "Balanced",
      "description": "Well-rounded default personality",
      "traits": {
        "openness": 0.7,
        "conscientiousness": 0.7,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.3
      }
    },
    {
      "id": "empathetic",
      "name": "Empathetic",
      "description": "High empathy, supportive tone",
      "traits": {
        "openness": 0.8,
        "agreeableness": 0.95
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personality/presets/apply',
      description: 'Apply preset',
      requestBody: `{
  "preset_id": "empathetic"
}`,
      params: [
        { name: 'preset_id', type: 'string', required: true, description: 'Preset identifier' },
      ],
      responseExample: `{
  "ok": true,
  "preset": "empathetic",
  "applied_traits": ["openness", "agreeableness"]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/conflicts',
      description: 'Detect conflicts',
      responseExample: `{
  "conflicts": [],
  "warnings": [
    "High agreeableness may reduce honest feedback"
  ]
}`,
    },
  ],
  personas: [
    {
      method: 'GET',
      path: '/consciousness/personas',
      description: 'List saved personas',
      responseExample: `{
  "personas": [
    {
      "id": "persona_abc123",
      "name": "Helpful Assistant",
      "active": true,
      "created_at": 1700000000,
      "traits": {
        "openness": 0.7,
        "agreeableness": 0.9
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personas/save',
      description: 'Save a persona',
      requestBody: `{
  "name": "My Persona",
  "traits": {
    "openness": 0.8,
    "conscientiousness": 0.7
  }
}`,
      responseExample: `{
  "id": "persona_def456",
  "name": "My Persona",
  "created_at": 1700000000
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personas/:id',
      description: 'Get a persona',
      params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }],
      responseExample: `{
  "id": "persona_abc123",
  "name": "Helpful Assistant",
  "traits": {
    "openness": 0.7,
    "agreeableness": 0.9
  },
  "communication_style": {
    "formality": 0.5,
    "empathy": 0.8
  },
  "active": true
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personas/:id/activate',
      description: 'Activate a persona',
      params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }],
      responseExample: `{
  "ok": true,
  "activated": "persona_abc123"
}`,
    },
    {
      method: 'DELETE',
      path: '/consciousness/personas/:id',
      description: 'Delete a persona',
      params: [{ name: 'id', type: 'string', required: true, description: 'Persona ID' }],
      responseExample: `{
  "ok": true,
  "deleted": "persona_abc123"
}`,
    },
  ],
  data: [
    {
      method: 'POST',
      path: '/consciousness/seed',
      description: 'Seed test data',
      params: [
        {
          name: 'count',
          type: 'number',
          required: false,
          description: 'Number of episodes to seed (default 10)',
        },
      ],
      responseExample: `{
  "ok": true,
  "seeded": 10
}`,
    },
  ],
}

export const METHOD_COLORS: Record<HttpMethod, string> = {
  GET: 'bg-success/15 text-success border-success/30',
  POST: 'bg-info/15 text-info border-info/30',
  PATCH: 'bg-warning/15 text-warning border-warning/30',
  DELETE: 'bg-destructive/15 text-destructive border-destructive/30',
  PUT: 'bg-violet-500/15 text-violet-400 border-violet-500/30',
}
