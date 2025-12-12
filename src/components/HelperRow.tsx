import React from 'react';
import { FaqPanel } from './FaqPanel';
import { AdditionalInfo } from '@/components/ui/additional-info';

interface HelperRowProps {
  onPickQuestion: (q: string) => void;
}

const HelperRow: React.FC<HelperRowProps> = ({ onPickQuestion }) => {
  return (
    <div className='w-full flex flex-col items-stretch gap-4 md:flex-row md:items-stretch md:justify-center md:gap-6'>
      {/* Left: FAQ section */}
      <div className='flex-1 flex min-w-0'>
        <FaqPanel onPickQuestion={onPickQuestion} />
      </div>

      {/* Right: Additional info - hidden on mobile */}
      <div className='hidden md:flex flex-1 min-w-0'>
        <AdditionalInfo />
      </div>
    </div>
  );
};

export { HelperRow };
