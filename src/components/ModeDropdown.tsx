import React, { useEffect, useRef, useState } from 'react';
import type { Mode } from './types';

interface ModeDropdownProps {
  mode: Mode;
  onModeChange: (m: Mode) => void;
}

const MODE_META: Record<Mode, { label: string; tagline: string }> = {
  beginner: {
    label: 'Beginner',
    tagline: 'Plain-language, step-by-step answers',
  },
  intermediate: {
    label: 'Intermediate',
    tagline: 'More detail, assumes some background',
  },
};

const ModeDropdown: React.FC<ModeDropdownProps> = ({ mode, onModeChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const current = MODE_META[mode];
  const rootRef = useRef<HTMLDivElement>(null);

  const handleSelect = (value: Mode) => {
    onModeChange(value);
    setIsOpen(false);
  };

  // Close when clicking outside
  useEffect(() => {
    const onDown = (e: MouseEvent | TouchEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('touchstart', onDown);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('touchstart', onDown);
    };
  }, []);

  return (
    <div ref={rootRef} className='relative inline-block w-[220px] mt-2'>
      <button
        type='button'
        onClick={() => setIsOpen((o) => !o)}
        className='w-full rounded-2xl cursor-pointer select-none bg-neutral-900 border border-neutral-800 shadow-xl shadow-black/40 px-3 py-2.5 flex items-center gap-3 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]'
      >
        <div className='flex-1 overflow-hidden text-left'>
          <p className='text-xs font-semibold text-neutral-50'>{current.label} mode</p>
          <p className='text-[11px] text-neutral-400 mt-0.5 truncate'>{current.tagline}</p>
        </div>
      </button>

      {isOpen && (
        <div
          className='
            absolute left-0 z-50 w-full
            opacity-100 translate-y-0 pointer-events-auto
            transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)]
            bottom-full mb-2
            md:bottom-auto md:top-full md:mt-2
          '
        >
          <div className='rounded-2xl border border-neutral-800 bg-neutral-950 shadow-2xl shadow-black/60'>
            <div className='px-2 pb-2 pt-2'>
              <div className='space-y-1'>
                {(Object.keys(MODE_META) as Mode[]).map((value) => {
                  const meta = MODE_META[value];
                  const isActive = value === mode;

                  return (
                    <button
                      key={value}
                      type='button'
                      onClick={() => handleSelect(value)}
                      className={`w-full flex flex-col items-start rounded-xl px-3 py-2 text-left transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] hover:bg-neutral-900 ${
                        isActive ? 'bg-neutral-900' : 'bg-transparent'
                      }`}
                    >
                      <span className='text-xs font-medium text-neutral-100'>
                        {meta.label} mode
                      </span>
                      <span className='text-[11px] text-neutral-400'>{meta.tagline}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export { ModeDropdown };
