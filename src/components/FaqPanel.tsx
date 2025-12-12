import React from 'react';

type FaqItem = {
  question: string;
};

const FAQ_ITEMS: FaqItem[] = [
  { question: 'Is creatine safe for beginners?' },
  { question: 'How much creatine should I take per day?' },
  { question: 'How much protein do I need per day to build muscle?' },
  { question: 'How many times per week should I train each muscle group?' },
];

interface FaqPanelProps {
  onPickQuestion: (q: string) => void;
}

const FaqPanel: React.FC<FaqPanelProps> = ({ onPickQuestion }) => {
  return (
    <div className='h-full flex-1 rounded-2xl border border-neutral-800 bg-neutral-950/70 p-4 md:p-5 shadow-[0_0_40px_rgba(0,0,0,0.6)]'>
      <div className='mb-3'>
        <h3 className='font-bold leading-tight text-white'>
          <span className='text-xl md:text-2xl lg:text-3xl'>Commonly asked questions</span>
        </h3>
      </div>

      <div className='space-y-2'>
        {FAQ_ITEMS.map((item) => (
          <button
            key={item.question}
            type='button'
            onClick={() => onPickQuestion(item.question)}
            className='w-full text-left rounded-xl bg-neutral-900/70 border border-neutral-800 px-3.5 py-2.5 md:px-4 md:py-3 transition-colors hover:border-neutral-600 hover:bg-neutral-900'
          >
            <span className='text-xs md:text-sm text-neutral-100'>{item.question}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export { FaqPanel };
