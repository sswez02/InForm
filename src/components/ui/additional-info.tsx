'use client';

import * as React from 'react';
import movieSrc from '@/assets/movie_h264.mp4';

export function AdditionalInfo() {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);

  React.useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    v.muted = true;
    v.playsInline = true;

    const tryPlay = () => v.play().catch(() => {});
    tryPlay();

    const onVis = () => {
      if (!document.hidden) tryPlay();
    };
    document.addEventListener('visibilitychange', onVis);

    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  const [cursor, setCursor] = React.useState({
    x: 0,
    y: 0,
    visible: false,
    variant: 'default' as 'default' | 'link',
  });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setCursor((prev) => ({
      ...prev,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      visible: true,
    }));
  };

  const handleMouseLeave = () => {
    setCursor((prev) => ({ ...prev, visible: false, variant: 'default' }));
  };

  const setLinkVariant = (isOverLink: boolean) => {
    setCursor((prev) => ({ ...prev, variant: isOverLink ? 'link' : 'default' }));
  };

  const cursorClasses =
    cursor.variant === 'link'
      ? 'h-6 w-6 border-neutral-100 bg-neutral-900/10'
      : 'h-6 w-6 border-neutral-100 bg-neutral-900/10';

  return (
    <section className='relative w-full h-full mt-1'>
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className='relative z-20 overflow-hidden h-full w-full rounded-2xl border border-neutral-800 bg-neutral-1000/70 shadow-[0_0_40px_rgba(0,0,0,0.6)] cursor-none'
      >
        {/* Video Background */}
        <div className='absolute inset-0 z-0'>
          <video
            ref={videoRef}
            src={movieSrc}
            autoPlay
            muted
            loop
            playsInline
            preload='auto'
            controls={false}
            disablePictureInPicture
            controlsList='nodownload nofullscreen noremoteplayback'
            className='absolute inset-0 h-full w-full object-cover scale-[1.1] pointer-events-none'
            onError={() => {
              const v = videoRef.current;
              if (!v) return;
              const err = v.error;
              console.log('VIDEO ERROR', {
                code: err?.code,
                message: err?.message,
                networkState: v.networkState,
                readyState: v.readyState,
                currentSrc: v.currentSrc,
              });
            }}
            onCanPlay={() => console.log('video can play')}
          />
          <div className='absolute inset-0 bg-black/40' />
          <div className='absolute inset-0 bg-gradient-to-b from-black/35 via-transparent to-black/55' />
        </div>

        <div className='relative z-10 flex h-full flex-col gap-5 px-5 py-6 md:flex-row md:items-center md:gap-8 md:px-7 md:py-7'>
          {/* LEFT: Pills */}
          <div className='relative mx-auto flex h-[210px] w-[260px] items-center justify-center md:h-[230px] md:w-[280px]'>
            <div className='flex flex-col items-center gap-4'>
              <a
                href='https://docs.google.com/document/d/1dl-clKpKe7fA6tvJ07uORNMl7Ms8Ja4p6CH0eqlXlNQ/edit?usp=sharing'
                target='_blank'
                rel='noreferrer'
                onMouseEnter={() => setLinkVariant(true)}
                onMouseLeave={() => setLinkVariant(false)}
                className='
                  rounded-full border border-neutral-700/70
                  bg-black px-4 py-1.5
                  text-[12px] md:text-[13px] text-neutral-100
                  transition-all duration-200 cursor-none
                  hover:border-neutral-300/80 hover:bg-black/90
                  hover:shadow-[0_0_25px_rgba(255,255,255,0.12)]
                '
              >
                How it works
              </a>

              <a
                href='https://docs.google.com/document/d/1LcseJ-41U6HMBc1nJWliYuqRzL9GHk8y0tx9JGYFYFw/edit?usp=sharing'
                target='_blank'
                rel='noreferrer'
                onMouseEnter={() => setLinkVariant(true)}
                onMouseLeave={() => setLinkVariant(false)}
                className='
                  rounded-full border border-neutral-700/70
                  bg-black px-4 py-1.5
                  text-[12px] md:text-[13px] text-neutral-100
                  transition-all duration-200 cursor-none
                hover:border-neutral-300/80 hover:bg-black/90
                  hover:shadow-[0_0_25px_rgba(255,255,255,0.12)]
                '
              >
                Learn about AI
              </a>
            </div>
          </div>

          {/* RIGHT: Title */}
          <div className='flex flex-1 flex-col justify-center'>
            <h3 className='font-bold text-white'>
              <span
                className='
                  inline
                  bg-black
                  text-white
                  px-3 py-1
                  box-decoration-clone
                  [-webkit-box-decoration-break:clone]
                  text-xl md:text-2xl lg:text-3xl
                  leading-[1.32]
                '
              >
                Additional info about InForm
              </span>
            </h3>
          </div>
        </div>

        {/* Custom cursor circle */}
        {cursor.visible && (
          <div
            className={`pointer-events-none absolute z-50 rounded-full border mix-blend-screen backdrop-blur-[1px] transition-[width,height,border-color,background-color] duration-150 ${cursorClasses}`}
            style={{ left: cursor.x, top: cursor.y, transform: 'translate(-50%, -50%)' }}
          />
        )}
      </div>
    </section>
  );
}
