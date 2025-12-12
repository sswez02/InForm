'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';

import { ModeDropdown } from './ModeDropdown';
import arrowImage from '@/assets/arrow.png';

interface PromptInputProps {
  onSubmit: (value: string) => void;
  variant?: 'hero' | 'chat';
  mode: 'beginner' | 'intermediate';
  onModeChange: (m: 'beginner' | 'intermediate') => void;
  value?: string;
  onChangeValue?: (v: string) => void;
}

const PromptInput: React.FC<PromptInputProps> = ({
  onSubmit,
  variant = 'hero',
  mode,
  onModeChange,
  value,
  onChangeValue,
}) => {
  const [internalValue, setInternalValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const isControlled = value !== undefined;
  const currentValue = isControlled ? value : internalValue;

  const setValue = (v: string) => {
    if (isControlled) onChangeValue?.(v);
    else setInternalValue(v);
  };

  const autoGrow = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = '0px';
    const next = Math.min(el.scrollHeight, 128);
    el.style.height = `${next}px`;
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const text = currentValue.trim();
    if (!text) return;
    onSubmit(text);
    setValue('');
    const el = textareaRef.current;
    if (el) el.style.height = '40px';
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  const canSend = currentValue.trim().length > 0;

  const placeholder =
    variant === 'hero'
      ? 'Ask anything about your training or nutrition...'
      : 'Ask a follow-up question...';

  useEffect(() => {
    autoGrow();
  }, [currentValue]);

  return (
    <form
      onSubmit={handleSubmit}
      className='
        h-full bg-neutral-950 border border-neutral-800 rounded-[30px]
        px-3 pt-3 pb-4 min-h-[96px]
        md:px-4 md:pt-4 md:pb-5 md:min-h-[110px]
        flex flex-col gap-3
        shadow-[0_0_0_1px_rgba(255,255,255,0.03)]
      '
    >
      <div className='flex items-start gap-3'>
        <textarea
          ref={textareaRef}
          rows={1}
          value={currentValue}
          onChange={(e) => setValue(e.target.value)}
          onInput={autoGrow}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className='flex-1 bg-transparent text-sm text-neutral-50 outline-none resize-none max-h-32 placeholder:text-neutral-500 pt-1.5'
        />
      </div>

      <div className='flex items-center justify-between gap-3'>
        <ModeDropdown mode={mode} onModeChange={onModeChange} />

        <div className='flex items-center gap-2 mt-2 md:mt-6'>
          <button
            type='submit'
            disabled={!canSend}
            className='h-9 w-9 rounded-full flex items-center justify-center text-black disabled:opacity-40 disabled:bg-neutral-700 disabled:text-neutral-300'
            style={{
              minWidth: '40px',
              minHeight: '40px',
              backgroundImage: `url(${arrowImage})`,
              backgroundSize: '30%',
              backgroundPosition: 'center',
              backgroundRepeat: 'no-repeat',
              backgroundColor: canSend ? 'white' : '#c7c5c5',
              filter: canSend ? 'invert(0)' : 'invert(1)',
            }}
          />
        </div>
      </div>
    </form>
  );
};

export { PromptInput };
