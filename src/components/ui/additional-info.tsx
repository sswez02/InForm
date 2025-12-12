'use client';

import * as React from 'react';

export function AdditionalInfo() {
  const containerRef = React.useRef<HTMLDivElement | null>(null);

  const [cursor, setCursor] = React.useState({
    x: 0,
    y: 0,
    visible: false,
    variant: 'default' as 'default' | 'link',
  });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCursor((prev) => ({
      ...prev,
      x,
      y,
      visible: true,
    }));
  };

  const handleMouseLeave = () => {
    setCursor((prev) => ({
      ...prev,
      visible: false,
      variant: 'default',
    }));
  };

  const setLinkVariant = (isOverLink: boolean) => {
    setCursor((prev) => ({
      ...prev,
      variant: isOverLink ? 'link' : 'default',
    }));
  };

  const cursorClasses =
    cursor.variant === 'link'
      ? 'h-5 w-5 border-neutral-100 bg-neutral-100/10'
      : 'h-5 w-5 border-neutral-100 bg-neutral-100/10';
  return (
    <section className='relative w-full h-full mt-1'>
      {/* Container*/}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className='
          relative z-20 overflow-hidden h-full w-full
          rounded-2xl border border-neutral-800
          bg-neutral-1000/70 shadow-[0_0_40px_rgba(0,0,0,0.6)]
          cursor-none
        '
      >
        <div className='flex h-full flex-col gap-5 px-5 py-6 md:flex-row md:items-center md:gap-8 md:px-7 md:py-7'>
          {/* LEFT: Pills */}
          <div className='relative mx-auto flex h-[210px] w-[260px] items-center justify-center md:h-[230px] md:w-[280px]'>
            <div className='flex flex-col items-center gap-4'>
              <a
                href='#'
                onMouseEnter={() => setLinkVariant(true)}
                onMouseLeave={() => setLinkVariant(false)}
                className='
                  rounded-full border border-neutral-700 
                  bg-neutral-900/80 px-4 py-1.5
                  text-[12px] md:text-[13px] text-neutral-100 shadow-md
                  transition-all duration-200 cursor-none
                  hover:-translate-y-0.5 hover:border-neutral-300 hover:bg-neutral-900
                  hover:shadow-[0_0_25px_rgba(255,255,255,0.15)]
                '
              >
                How it works
              </a>
              <a
                href='#'
                onMouseEnter={() => setLinkVariant(true)}
                onMouseLeave={() => setLinkVariant(false)}
                className='
                  rounded-full border border-neutral-700 
                  bg-neutral-900/80 px-4 py-1.5
                  text-[12px] md:text-[13px] text-neutral-100 shadow-md
                  transition-all duration-200 cursor-none
                  hover:-translate-y-0.5 hover:border-neutral-300 hover:bg-neutral-900
                  hover:shadow-[0_0_25px_rgba(255,255,255,0.15)]
                '
              >
                Learn about AI
              </a>
            </div>
          </div>

          {/* RIGHT: Title */}
          <div className='flex flex-1 flex-col justify-center'>
            <h3 className='font-bold leading-tight text-white'>
              <span className='text-xl md:text-2xl lg:text-3xl'>Additional info about InForm</span>
            </h3>
          </div>
        </div>

        {/* Custom cursor circle (only inside card) */}
        {cursor.visible && (
          <div
            className={`
              pointer-events-none absolute rounded-full border
              mix-blend-screen backdrop-blur-sm
              transition-[width,height,border-color,background-color] duration-150
              ${cursorClasses}
            `}
            style={{
              left: cursor.x,
              top: cursor.y,
              transform: 'translate(-50%, -50%)',
            }}
          />
        )}
      </div>
    </section>
  );
}
