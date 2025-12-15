import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Mode } from './types';

interface ModeDropdownProps {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  openDirection?: 'up' | 'down';
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

type MenuPos = { left: number; top: number; bottom: number; width: number };

const ModeDropdown: React.FC<ModeDropdownProps> = ({
  mode,
  onModeChange,
  openDirection = 'up',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<MenuPos | null>(null);

  const current = MODE_META[mode];

  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const measureAndSetMenuPos = () => {
    const btn = buttonRef.current;
    if (!btn) return;
    const r = btn.getBoundingClientRect();
    setMenuPos({ left: r.left, top: r.top, bottom: r.bottom, width: r.width });
  };

  const handleSelect = (value: Mode) => {
    if (value === mode) {
      setIsOpen(false);
      return;
    }
    onModeChange(value);
    setIsOpen(false);
  };

  useEffect(() => {
    const onDown = (e: MouseEvent | TouchEvent) => {
      const t = e.target as Node;
      const inRoot = rootRef.current?.contains(t);
      const inMenu = menuRef.current?.contains(t);
      if (!inRoot && !inMenu) setIsOpen(false);
    };

    document.addEventListener('mousedown', onDown);
    document.addEventListener('touchstart', onDown);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('touchstart', onDown);
    };
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const onReflow = () => measureAndSetMenuPos();

    window.addEventListener('resize', onReflow);
    window.addEventListener('scroll', onReflow, true);

    measureAndSetMenuPos();

    return () => {
      window.removeEventListener('resize', onReflow);
      window.removeEventListener('scroll', onReflow, true);
    };
  }, [isOpen]);

  const toggle = () => {
    if (isOpen) {
      setIsOpen(false);
      return;
    }
    measureAndSetMenuPos();
    setIsOpen(true);
  };

  return (
    <div ref={rootRef} className='relative inline-block w-[220px] mt-2'>
      <button
        ref={buttonRef}
        type='button'
        onClick={toggle}
        className='w-full rounded-2xl cursor-pointer select-none bg-neutral-900 border border-neutral-800 shadow-xl shadow-black/40 px-3 py-2.5 flex items-center gap-3 transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]'
      >
        <div className='flex-1 overflow-hidden text-left'>
          <p className='text-xs font-semibold text-neutral-50'>{current.label} mode</p>
          <p className='text-[11px] text-neutral-400 mt-0.5 truncate'>{current.tagline}</p>
        </div>
      </button>

      {isOpen &&
        menuPos &&
        createPortal(
          <div
            ref={menuRef}
            className='fixed z-[9999] pointer-events-none'
            style={{
              left: menuPos.left,
              top: openDirection === 'up' ? menuPos.top : menuPos.bottom,
              width: menuPos.width,
            }}
          >
            <div
              className={
                openDirection === 'up' ? 'pointer-events-auto' : 'mt-2 pointer-events-auto'
              }
              style={
                openDirection === 'up' ? { transform: 'translateY(calc(-100% - 8px))' } : undefined
              }
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
          </div>,
          document.body
        )}
    </div>
  );
};

export { ModeDropdown };
