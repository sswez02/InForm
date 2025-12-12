'use client';

import React, { useRef, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

const Pointer = ({ ...props }: React.SVGProps<SVGSVGElement>) => {
  return (
    <svg
      stroke='currentColor'
      fill='currentColor'
      strokeWidth='1'
      strokeLinecap='round'
      strokeLinejoin='round'
      viewBox='0 0 16 16'
      height='1em'
      width='1em'
      xmlns='http://www.w3.org/2000/svg'
      {...props}
    >
      <path d='M14.082 2.182a.5.5 0 0 1 .103.557L8.528 15.467a.5.5 0 0 1-.917-.007L5.57 10.694.803 8.652a.5.5 0 0 1-.006-.916l12.728-5.657a.5.5 0 0 1 .556.103z'></path>
    </svg>
  );
};

export function PointerHighlight({
  children,
  rectangleClassName,
  pointerClassName,
  containerClassName,
}: {
  children: React.ReactNode; // content
  rectangleClassName?: string; // border box
  pointerClassName?: string; // cursor icon
  containerClassName?: string; // wrapper
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!containerRef.current) return;

    const updateSize = () => {
      if (!containerRef.current) return;
      const { width, height } = containerRef.current.getBoundingClientRect();
      setDimensions({ width, height });
    };

    updateSize();

    const resizeObserver = new ResizeObserver(() => updateSize());
    resizeObserver.observe(containerRef.current);

    return () => {
      if (containerRef.current) {
        resizeObserver.unobserve(containerRef.current);
      }
    };
  }, []);

  // Box padding around the text
  const paddingX = 10;
  const paddingY = 6;

  // Pointer position - bottom-right
  const pointerLeft = dimensions.width + paddingX * 2 - 14;
  const pointerTop = dimensions.height + paddingY * 2 - 14;

  return (
    <div className={cn('relative w-fit inline-flex', containerClassName)} ref={containerRef}>
      {children}

      {dimensions.width > 0 && dimensions.height > 0 && (
        <>
          {/* Rectangle around the text */}
          <div
            className={cn(
              'pointer-events-none absolute border border-neutral-800',
              rectangleClassName
            )}
            style={{
              top: -paddingY,
              left: -paddingX,
              width: dimensions.width + paddingX * 2,
              height: dimensions.height + paddingY * 2,
            }}
          />

          {/* Static pointer in the bottom-right corner */}
          <div
            className={cn('pointer-events-none absolute', pointerClassName)}
            style={{
              transform: 'rotate(-90deg)',
              top: pointerTop,
              left: pointerLeft,
            }}
          >
            <Pointer className='h-5 w-5' />
          </div>
        </>
      )}
    </div>
  );
}
