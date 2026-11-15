/**
 * Static data for the Agents page: the tool catalogue and the starter-agent
 * templates offered by the creator card.
 */
export const AVAILABLE_TOOLS = [
  'code_execution',
  'file_read',
  'file_search',
  'web_search',
  'knowledge_retrieval',
  'image_analysis',
  'data_analysis',
  'citation',
]

export const AGENT_TEMPLATES = [
  {
    name: 'Researcher',
    desc: 'Finds information from the web and documents',
    instructions:
      'You are a thorough researcher. Search for accurate, up-to-date information and present findings clearly with sources.',
    tools: ['web_search', 'knowledge_retrieval'],
  },
  {
    name: 'Coder',
    desc: 'Writes and debugs code',
    instructions:
      'You are an expert programmer. Write clean, efficient code. Explain your approach. Debug errors methodically.',
    tools: ['code_execution', 'file_read'],
  },
  {
    name: 'Analyst',
    desc: 'Analyzes data and generates insights',
    instructions:
      'You are a data analyst. Examine datasets, identify patterns, compute statistics, and present clear visualizations and insights.',
    tools: ['data_analysis', 'file_read'],
  },
  {
    name: 'Writer',
    desc: 'Creates structured written content',
    instructions:
      'You are a skilled writer. Produce clear, well-organized content. Adapt tone to the audience. Ensure accuracy.',
    tools: ['knowledge_retrieval'],
  },
  {
    name: 'Vision Assistant',
    desc: 'Analyzes images and visual data',
    instructions:
      'You are a vision expert. Analyze images carefully, describe what you see, identify objects, text, and patterns.',
    tools: ['image_analysis'],
  },
  {
    name: 'Full Stack',
    desc: 'Handles all aspects of a task',
    instructions:
      'You are a versatile full-stack assistant. Use whatever tools are needed to complete the task: search, code, analyze, or write.',
    tools: [
      'web_search',
      'code_execution',
      'file_read',
      'knowledge_retrieval',
      'data_analysis',
      'file_search',
    ],
  },
]
