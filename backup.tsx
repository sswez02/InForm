'use client';

import React, { useEffect } from 'react';
import { useMotionValue, useMotionValueEvent } from 'framer-motion';
import type { MotionValue } from 'framer-motion';

import { ScrollText } from '@/components/ui/scroll-text';
import { PointerHighlight } from '@/components/ui/pointer-highlight';

const SCRIPT_LINES = [
  'Scroll to reveal..',
  'We live in an age where information is everywhere,',
  'yet contradictions and confusion persist, even among experts.',
  'To help you navigate the sea of knowledge, InForm grounds every answer in real research.',
];

const HeroScript = ({ progress }: { progress: MotionValue<number> }) => {
  const lineCount = SCRIPT_LINES.length;

  const lineProgress = useMotionValue(0);
  const [activeIndex, setActiveIndex] = React.useState(0);

  useMotionValueEvent(progress, 'change', (p: number) => {
    if (lineCount === 0) return;

    const clamped = Math.max(0, Math.min(0.9999, p));

    const segmentSize = 1 / lineCount;
    const index = Math.floor(clamped / segmentSize); // line number
    const base = index * segmentSize;
    const within = (clamped - base) / segmentSize; // progress within line

    setActiveIndex(index);
    lineProgress.set(within);
  });

  return (
    <div className='mt-1'>
      <ScrollText text={SCRIPT_LINES[activeIndex]} progress={lineProgress} />
    </div>
  );
};

const AnimatedTitleBlock = () => {
  const progress = useMotionValue(0);

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      const current = progress.get();
      const TOTAL_SCROLL_PX = 10000;

      const next = Math.max(0, Math.min(1, current + e.deltaY / TOTAL_SCROLL_PX));
      progress.set(next);
    };

    window.addEventListener('wheel', handleWheel, { passive: true });
    return () => window.removeEventListener('wheel', handleWheel);
  }, [progress]);

  return (
    <div className='relative mt-10 md:mt-20 flex flex-col items-center justify-center gap-2 md:gap-3 h-[6.5rem] md:h-[8.5rem]'>
      <h1 className='text-3xl sm:text-4xl md:text-5xl lg:text-[2.7rem] font-semibold tracking-tight text-center text-neutral-400'>
        {/* Title */}
        <span className='text-neutral-500'>Stay informed, be</span>{' '}
        <PointerHighlight
          rectangleClassName='rounded-[4px] border-neutral-500'
          pointerClassName='text-white'
          containerClassName='inline-flex ml-1'
        >
          <span className='relative z-10 font-semibold text-white'>InForm.</span>
        </PointerHighlight>
      </h1>

      {/* Hide scroll text on mobile */}
      <div className='hidden md:block'>
        <HeroScript progress={progress} />
      </div>
    </div>
  );
};

export default AnimatedTitleBlock;
