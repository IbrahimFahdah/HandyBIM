export interface Point2D {
  x: number;
  y: number;
}

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface HandControlOptions {
  /** Video element already playing a webcam (or other) stream. */
  videoElement: HTMLVideoElement;
  /** Maximum number of hands to track. Default 2. */
  maxHands?: number;
  /** Exponential smoothing weight for new landmark samples, 0..1. Default 0.2. */
  smoothing?: number;
  /** Flip the x axis, for a mirrored webcam preview. Default true. */
  mirrorX?: boolean;
  /** Override the MediaPipe HandLandmarker model asset URL. */
  modelAssetPath?: string;
  /** Override the MediaPipe tasks-vision WASM base URL. */
  wasmBasePath?: string;
}

/** Gesture state machine values, per the hand-control spec's priority order. */
export enum GestureState {
  IDLE = 'IDLE',
  HAND_VISIBLE = 'HAND_VISIBLE',
  POINTING = 'POINTING',
  PINCHING = 'PINCHING',
  ROTATING = 'ROTATING',
  PANNING = 'PANNING',
  TWO_HAND_ZOOM = 'TWO_HAND_ZOOM',
  SECTION_ACTIVE = 'SECTION_ACTIVE',
  PAUSE_TOGGLE = 'PAUSE_TOGGLE',
  RESET_VIEW = 'RESET_VIEW',
}

export interface GestureEventMap {
  [key: string]: unknown;
  rotate: { dx: number; dy: number };
  zoom: { scale: number };
  pan: { dx: number; dy: number };
  pointer: { x: number; y: number };
  select: { x: number; y: number };
  sectionStart: undefined;
  sectionMove: { delta: number };
  sectionRotate: { deltaAngle: number };
  sectionEnd: undefined;
  pauseToggle: undefined;
  resetView: undefined;
  landmarks: Landmark[][];
}

export type GestureEventName = keyof GestureEventMap;
