'use client';

import React, { useState } from 'react';
import type { Mode } from '../types';
import { PromptInput } from '../PromptInput';
import { HelperRow } from '../HelperRow';
import AnimatedTitleBlock from '../AnimatedTitleBlock';

interface LandingViewProps {
  onSubmit: (value: string) => void;
  mode: Mode;
  onModeChange: (m: Mode) => void;
}

const LandingView: React.FC<LandingViewProps> = ({ onSubmit, mode, onModeChange }) => {
  const [heroPrompt, setHeroPrompt] = useState('');

  return (
    <div className='flex-1 relative px-3 md:px-4 pb-[calc(env(safe-area-inset-bottom)+12px)] md:pb-0'>
      <div
        className='
          w-full max-w-5xl
          mx-auto
          flex flex-col items-center gap-5 md:gap-6
          mt-6 md:mt-0
          md:absolute md:left-1/2 md:top-[40vh]
          md:-translate-x-1/2 md:-translate-y-1/2
        '
      >
        <AnimatedTitleBlock />

        <div className='w-full md:transform md:translate-y-10'>
          <HelperRow onPickQuestion={setHeroPrompt} />
        </div>

        <div className='w-full max-w-md md:max-w-none md:w-[40vw] -mt-2 md:mt-[10vh]'>
          <PromptInput
            onSubmit={onSubmit}
            variant='hero'
            mode={mode}
            onModeChange={onModeChange}
            value={heroPrompt}
            onChangeValue={setHeroPrompt}
          />
        </div>
      </div>
    </div>
  );
};

export default LandingView;
