'use client';

import * as React from 'react';
import { motion, useTransform, MotionValue } from 'framer-motion';

export interface ScrollTextProps {
  text: string;
  progress: MotionValue<number>;
}

interface WordProps {
  children: string;
  progress: MotionValue<number>;
  range: number[];
}

const Word: React.FC<WordProps> = ({ children, progress, range }) => {
  const opacity = useTransform(progress, range, [0, 1]);

  return (
    <span className='relative inline-block mr-2 text-base sm:text-lg md:text-2xl font-semibold'>
      <span className='text-neutral-800'>{children}</span>
      <motion.span className='absolute inset-0 text-neutral-50' style={{ opacity }}>
        {children}
      </motion.span>
    </span>
  );
};

export const ScrollText: React.FC<ScrollTextProps> = ({ text, progress }) => {
  const words = text.split(' ').filter(Boolean);
  const totalWords = words.length || 1;

  return (
    <div className='inline-block cursor-default'>
      <p className='flex flex-wrap md:flex-nowrap justify-center text-center leading-tight px-2 md:px-4 md:whitespace-nowrap'>
        {words.map((word, i) => {
          const start = i / totalWords;
          const end = (i + 1) / totalWords;

          return (
            <Word key={`${word}-${i}`} progress={progress} range={[start, end]}>
              {word}
            </Word>
          );
        })}
      </p>
    </div>
  );
};
