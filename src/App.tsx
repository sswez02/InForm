'use client';

import { useState } from 'react';
import type { Message, Confidence } from './components/types';

import LandingView from './components/views/LandingView';
import ChatView from './components/views/ChatView';

type CitationRef = {
  index: number;
  study_id: number;
  title?: string | null;
};

type AskResponse = {
  answer: string;
  mode: string;
  query: string;
  backend: 'baseline' | 'llm';
  citations: CitationRef[];
  confidence: Confidence;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [mode, setMode] = useState<'beginner' | 'intermediate'>('beginner');
  const [isThinking, setIsThinking] = useState(false);

  const hasChat = messages.length > 0;

  const handleSend = async (value: string) => {
    const text = value.trim();
    if (!text) return;

    const userMessage: Message = {
      id: Date.now(),
      author: 'user',
      content: text,
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsThinking(true);

    try {
      const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          query: text,
          use_llm: true,
          top_k_passages: 10,
          max_studies: 3,
        }),
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const data: AskResponse = await res.json();

      let content = data.answer;

      if (data.citations.length > 0) {
        const refs = data.citations
          .map((c) => `[${c.index}] ${c.title ?? `Study ${c.study_id}`}`)
          .join('\n');
        content += `\n\nReferences:\n${refs}`;
      }

      const assistantMessage: Message = {
        id: Date.now() + 1,
        author: 'assistant',
        content,
        confidence: data.confidence,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (e: any) {
      const assistantMessage: Message = {
        id: Date.now() + 1,
        author: 'assistant',
        content: `Error talking to backend: ${e?.message ?? 'Something went wrong'}`,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className='h-[100dvh] md:h-screen bg-black text-neutral-50 flex flex-col overflow-hidden'>
      <main className='flex-1 flex flex-col overflow-hidden'>
        {!hasChat ? (
          <LandingView onSubmit={handleSend} mode={mode} onModeChange={setMode} />
        ) : (
          <ChatView
            messages={messages}
            onSubmit={handleSend}
            mode={mode}
            onModeChange={setMode}
            isThinking={isThinking}
          />
        )}
      </main>
    </div>
  );
}
