// Minimal Web Audio API sound manager for interaction feedback clicks.
export class AudioManager {
  constructor() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = null;
    this.isInitialized = false;
    this.lastClickTime = 0;
    this.clickInterval = 200; // ms between clicks, keeps rapid pinch updates from buzzing

    if (AudioContext) {
      try {
        this.audioCtx = new AudioContext();
        this.isInitialized = true;
      } catch (e) {
        console.error('Error creating AudioContext:', e);
      }
    } else {
      console.warn('Web Audio API is not supported in this browser.');
    }
  }

  // Browsers suspend AudioContext until a user gesture resumes it.
  resumeContext() {
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume().catch((e) => console.error('Error resuming AudioContext:', e));
    }
  }

  playInteractionClickSound() {
    if (!this.isInitialized || !this.audioCtx || this.audioCtx.state !== 'running') return;

    const now = this.audioCtx.currentTime;
    if (now - this.lastClickTime < this.clickInterval / 1000) return;
    this.lastClickTime = now;

    const oscillator = this.audioCtx.createOscillator();
    const gainNode = this.audioCtx.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(this.audioCtx.destination);

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(1200, now);
    oscillator.frequency.exponentialRampToValueAtTime(600, now + 0.01);

    const clickVolume = 0.08;
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(clickVolume, now + 0.003);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.005);

    oscillator.start(now);
    oscillator.stop(now + 0.005);
  }
}
