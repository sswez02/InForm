'use client';

import React, { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import type { Message } from '../types';
import type { Variants } from 'framer-motion';

import { PromptInput } from '../PromptInput';

/**
 * Turn plain-text answer + references into simple HTML
 */
function formatAssistantContent(content: string): string {
  const escape = (s: string) =>
    // Prevent HTML code
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const formatInline = (s: string) => {
    let h = escape(s);
    // Bold: **text**
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    return h;
  };

  const lines = content.split('\n');
  const parts: string[] = [];
  let inList = false;

  // Bullet points
  for (const rawLine of lines) {
    const line = rawLine ?? '';
    const bulletMatch = line.match(/^\s*[\*\-]\s+(.*)$/);

    if (bulletMatch) {
      const itemText = bulletMatch[1];

      if (!inList) {
        parts.push("<ul class='pl-5 space-y-1' style='list-style-type: disc;'>");
        inList = true;
      }

      parts.push(`<li>${formatInline(itemText)}</li>`);
    } else {
      if (inList) {
        parts.push('</ul>');
        inList = false;
      }

      if (line.trim().length === 0) {
        parts.push('<br />');
      } else {
        parts.push(`<p>${formatInline(line)}</p>`);
      }
    }
  }

  if (inList) {
    parts.push('</ul>');
  }

  return parts.join('');
}

/* Confidence bar */

const ConfidenceBar: React.FC<{ confidence: { value: number; label: string } }> = ({
  confidence,
}) => {
  const percent = Math.max(0, Math.min(100, confidence.value));
  const numericText = `${percent.toFixed(0)}%`;

  return (
    <div className='mb-3'>
      <div className='flex items-baseline gap-3'>
        <span className='text-2xl font-semibold text-neutral-50 leading-none'>{numericText}</span>
        <span className='text-xs text-neutral-400 tracking-wide mt-[2px]'>confidence</span>
      </div>

      <div className='mt-2 h-2 w-full rounded-full bg-neutral-900 overflow-hidden'>
        <div
          className='h-full w-full origin-left'
          style={{
            transform: `scaleX(${percent / 100})`,
            willChange: 'transform',
            backfaceVisibility: 'hidden',
            transformOrigin: 'left',
            backgroundImage:
              'linear-gradient(to right, ' +
              '#34d399 0%, #34d399 33.33%, ' +
              '#a3e635 33.33%, #a3e635 66.66%, ' +
              '#facc15 66.66%, #facc15 100%)',
          }}
        />
      </div>
    </div>
  );
};

/* Animated assistant content */

const AnimatedAssistantContent: React.FC<{ content: string }> = ({ content }) => {
  const blocks = content
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter((b) => b.length > 0);

  const containerVariants: Variants = {
    hidden: {},
    visible: { transition: { staggerChildren: 0.12 } },
  };

  const lineVariants: Variants = {
    hidden: { opacity: 0, y: 4, filter: 'blur(2px)' },
    visible: {
      opacity: 1,
      y: 0,
      filter: 'blur(0px)',
      transition: { duration: 0.35 },
    },
  };

  if (blocks.length === 0) {
    return (
      <div
        className='mt-2 text-sm leading-relaxed'
        dangerouslySetInnerHTML={{ __html: formatAssistantContent(content) }}
      />
    );
  }

  return (
    <motion.div
      className='mt-2 text-sm leading-relaxed space-y-2'
      variants={containerVariants}
      initial='hidden'
      animate='visible'
    >
      {blocks.map((block, idx) => (
        <motion.div
          key={idx}
          variants={lineVariants}
          className='whitespace-pre-wrap'
          dangerouslySetInnerHTML={{ __html: formatAssistantContent(block) }}
        />
      ))}
    </motion.div>
  );
};

/* Message bubble */

const MessageBubble: React.FC<{
  author: 'user' | 'assistant';
  content: string;
  confidence?: { value: number; label: string };
}> = ({ author, content, confidence }) => {
  const isUser = author === 'user';

  if (isUser) {
    return (
      <div className='flex justify-end'>
        <div className='max-w-[85%] md:max-w-[60%] rounded-2xl bg-neutral-900 border border-neutral-700 px-4 py-2.5 text-sm leading-relaxed text-neutral-50 shadow-sm'>
          <div className='mb-1 text-[11px] text-neutral-400'>You</div>
          <p className='whitespace-pre-wrap'>{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className='flex justify-start'>
      <div className='max-w-[92%] md:max-w-[80%] text-sm leading-relaxed text-neutral-100'>
        {confidence && <ConfidenceBar confidence={confidence} />}
        <AnimatedAssistantContent content={content} />
      </div>
    </div>
  );
};

/* “Thinking…” text */

const ThinkingText: React.FC<{ text: string }> = ({ text }) => (
  <motion.span
    className='inline-flex text-sm md:text-base font-medium bg-gradient-to-r from-neutral-500 via-neutral-50 to-neutral-500 bg-[length:200%_100%] bg-clip-text text-transparent'
    animate={{ backgroundPositionX: ['200%', '-200%'] }}
    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
    style={{ backgroundPositionX: '200%' }}
  >
    {text}
  </motion.span>
);

interface ChatViewProps {
  messages: Message[];
  onSubmit: (value: string) => void;
  mode: 'beginner' | 'intermediate';
  onModeChange: (m: 'beginner' | 'intermediate') => void;
  isThinking?: boolean;
}

const ChatView: React.FC<ChatViewProps> = ({
  messages,
  onSubmit,
  mode,
  onModeChange,
  isThinking,
}) => {
  // Inner scroll container
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages.length, isThinking]);

  return (
    <section className='flex-1 flex flex-col overflow-hidden'>
      <div ref={scrollRef} className='flex-1 overflow-y-auto custom-scroll scroll-smooth'>
        <div className='max-w-3xl mx-auto px-4 md:px-0 py-4 md:py-8 space-y-4 md:space-y-6'>
          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              author={message.author}
              content={message.content}
              confidence={message.confidence}
            />
          ))}

          {isThinking && (
            <div className='flex justify-start'>
              <div className='max-w-[92%] md:max-w-[80%] text-base md:text-lg leading-relaxed text-neutral-100'>
                <ThinkingText text='Thinking with studies…' />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className='bg-black px-3 md:px-12 py-3 pb-[calc(env(safe-area-inset-bottom)+12px)] md:pb-3'>
        <div className='max-w-3xl mx-auto'>
          <PromptInput onSubmit={onSubmit} variant='chat' mode={mode} onModeChange={onModeChange} />
        </div>
      </div>
    </section>
  );
};

export default ChatView;
