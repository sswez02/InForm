export type Mode = 'beginner' | 'intermediate';

export type Confidence = {
  value: number;
  label: 'low' | 'medium' | 'high';
};

export type AskResponse = {
  answer: string;
  mode: string;
  query: string;
  backend: 'baseline' | 'llm';
  citations: { index: number; study_id: number; title?: string | null }[];
  confidence: Confidence;
};

export type Author = 'user' | 'assistant';

export interface Message {
  id: number;
  author: Author;
  content: string;
  confidence?: Confidence;
}
