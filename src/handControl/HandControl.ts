import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { TypedEventEmitter } from './EventEmitter';
import { GestureState } from './types';
import type { GestureEventMap, HandControlOptions, Landmark, Point2D } from './types';
import { angleDelta, applyDeadZone, distance2D, handAngle, isFingerExtended, smooth } from './mathUtils';

const MEDIAPIPE_VERSION = '0.10.14';
const DEFAULT_WASM_BASE_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const DEFAULT_MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Pinch distance thresholds (normalized landmark units), with hysteresis so
// the signal doesn't chatter right at the boundary.
const PINCH_START = 0.045;
const PINCH_END = 0.065;

// Frame-based gesture confirmation, per spec: most gestures need 3
// consecutive frames to activate and tolerate 3 missing frames before
// cancelling; section mode uses a slower, more deliberate 5.
const CONFIRM_FRAMES = 3;
const CANCEL_FRAMES = 3;
const SECTION_CONFIRM_FRAMES = 5;
const SECTION_CANCEL_FRAMES = 5;
const UTILITY_CONFIRM_FRAMES = 8;
const HAND_LOST_CANCEL_FRAMES = 3;

const DEAD_ZONE = 0.003;

interface HandFrame {
  landmarks: Landmark[];
  center: Point2D; // middle-finger MCP, a stable proxy for "hand position"
  indexTip: Point2D;
  pinchDistance: number;
  extended: { index: boolean; middle: boolean; ring: boolean; pinky: boolean };
  thumbExtended: boolean;
}

interface GestureBaseline {
  handCenter?: Point2D;
  twoHandDistance?: number;
  handAngle?: number;
  handY?: number;
}

/**
 * Standalone webcam hand-gesture engine. Knows nothing about any particular
 * application — it only detects hands and emits normalized control deltas
 * (rotate / zoom / pan / pointer / select / section). What those deltas mean
 * to a 3D scene is entirely up to whoever listens.
 */
export class HandControl extends TypedEventEmitter<GestureEventMap> {
  private readonly videoElement: HTMLVideoElement;
  private readonly maxHands: number;
  private readonly smoothing: number;
  private readonly mirrorX: boolean;
  private readonly modelAssetPath: string;
  private readonly wasmBasePath: string;

  private handLandmarker: HandLandmarker | null = null;
  private running = false;
  private paused = false;
  private rafHandle: number | null = null;
  private lastVideoTime = -1;

  private smoothedLandmarks: (Landmark[] | null)[] = [null, null];
  private pinchActive: boolean[] = [false, false];
  private lastPointerPos: Point2D = { x: 0.5, y: 0.5 };

  private rawCandidateStreak: { candidate: GestureState; count: number } = {
    candidate: GestureState.IDLE,
    count: 0,
  };
  private confirmedState: GestureState = GestureState.IDLE;
  private missingStreak = 0;
  private framesWithoutHand = 0;
  private baseline: GestureBaseline = {};

  constructor(options: HandControlOptions) {
    super();
    this.videoElement = options.videoElement;
    this.maxHands = options.maxHands ?? 2;
    this.smoothing = options.smoothing ?? 0.2;
    this.mirrorX = options.mirrorX ?? true;
    this.modelAssetPath = options.modelAssetPath ?? DEFAULT_MODEL_ASSET_PATH;
    this.wasmBasePath = options.wasmBasePath ?? DEFAULT_WASM_BASE_PATH;
  }

  /** Current confirmed gesture state, for UI feedback. Not part of the event stream. */
  get state(): GestureState {
    return this.confirmedState;
  }

  get isPaused(): boolean {
    return this.paused;
  }

  pause(): void {
    if (this.paused) return;
    this.paused = true;
    this.resetTrackingState();
  }

  resume(): void {
    if (!this.paused) return;
    this.paused = false;
    this.lastVideoTime = -1;
    this.resetTrackingState();
  }

  reset(): void {
    this.resetTrackingState();
    this.transitionTo(GestureState.IDLE);
  }

  async start(): Promise<void> {
    if (this.running) return;
    const vision = await FilesetResolver.forVisionTasks(this.wasmBasePath);
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: this.modelAssetPath, delegate: 'GPU' },
      numHands: this.maxHands,
      runningMode: 'VIDEO',
    });
    this.running = true;
    this.lastVideoTime = -1;
    this.loop();
  }

  stop(): void {
    this.running = false;
    if (this.rafHandle !== null) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = null;
  }

  private loop = (): void => {
    if (!this.running) return;
    this.rafHandle = requestAnimationFrame(this.loop);
    this.tick();
  };

  private tick(): void {
    const video = this.videoElement;
    if (!this.handLandmarker || video.readyState < 2 || video.videoWidth === 0) return;
    const videoTime = video.currentTime;
    if (videoTime <= this.lastVideoTime) return;
    this.lastVideoTime = videoTime;

    const results = this.handLandmarker.detectForVideo(video, performance.now());
    this.processFrame(results.landmarks ?? []);
  }

  private processFrame(rawHands: Landmark[][]): void {
    if (this.paused) return;

    if (rawHands.length === 0) {
      this.emit('landmarks', []);
      this.framesWithoutHand++;
      if (this.framesWithoutHand >= HAND_LOST_CANCEL_FRAMES) {
        this.pinchActive = [false, false];
        this.transitionTo(GestureState.IDLE);
        this.rawCandidateStreak = { candidate: GestureState.IDLE, count: 0 };
      }
      return;
    }
    this.framesWithoutHand = 0;

    const hands = rawHands.slice(0, this.maxHands).map((lm, i) => this.buildHandFrame(lm, i));
    this.emit('landmarks', hands.map((hand) => hand.landmarks));
    this.lastPointerPos = hands[0].indexTip;

    hands.forEach((hand, i) => {
      const wasPinching = this.pinchActive[i];
      if (!wasPinching && hand.pinchDistance < PINCH_START) {
        this.pinchActive[i] = true;
        if (i === 0) this.emit('select', { x: this.lastPointerPos.x, y: this.lastPointerPos.y });
      } else if (wasPinching && hand.pinchDistance > PINCH_END) {
        this.pinchActive[i] = false;
      }
    });

    const rawCandidate = this.classify(hands);
    this.advanceDebouncer(rawCandidate);
    this.driveActiveState(hands);
  }

  private resetTrackingState(): void {
    this.smoothedLandmarks = [null, null];
    this.pinchActive = [false, false];
    this.lastPointerPos = { x: 0.5, y: 0.5 };
    this.rawCandidateStreak = { candidate: GestureState.IDLE, count: 0 };
    this.missingStreak = 0;
    this.framesWithoutHand = 0;
    this.baseline = {};
  }

  private buildHandFrame(rawLandmarks: Landmark[], handIndex: number): HandFrame {
    const previous = this.smoothedLandmarks[handIndex];
    const smoothed: Landmark[] = rawLandmarks.map((lm, i) => {
      const mirroredX = this.mirrorX ? 1 - lm.x : lm.x;
      if (!previous || previous.length !== rawLandmarks.length) {
        return { x: mirroredX, y: lm.y, z: lm.z };
      }
      return {
        x: smooth(previous[i].x, mirroredX, this.smoothing),
        y: smooth(previous[i].y, lm.y, this.smoothing),
        z: smooth(previous[i].z, lm.z, this.smoothing),
      };
    });
    this.smoothedLandmarks[handIndex] = smoothed;

    return {
      landmarks: smoothed,
      center: smoothed[9], // middle-finger MCP
      indexTip: smoothed[8],
      pinchDistance: distance2D(smoothed[4], smoothed[8]),
      extended: {
        index: isFingerExtended(smoothed, 8, 6),
        middle: isFingerExtended(smoothed, 12, 10),
        ring: isFingerExtended(smoothed, 16, 14),
        pinky: isFingerExtended(smoothed, 20, 18),
      },
      thumbExtended: isFingerExtended(smoothed, 4, 3),
    };
  }

  // Priority order per spec: SECTION_ACTIVE > TWO_HAND_ZOOM > ROTATING/PINCHING > PANNING > POINTING.
  private classify(hands: HandFrame[]): GestureState {
    const hand0 = hands[0];

    const hasExtendedFinger = hand0.extended.index || hand0.extended.middle || hand0.extended.ring || hand0.extended.pinky;
    if (hand0.thumbExtended && !hasExtendedFinger && !this.pinchActive[0]) return GestureState.PAUSE_TOGGLE;
    if (!hand0.thumbExtended && !hasExtendedFinger && !this.pinchActive[0]) return GestureState.RESET_VIEW;

    const isPeaceSign = hand0.extended.index && hand0.extended.middle && !hand0.extended.ring && !hand0.extended.pinky;
    if (isPeaceSign) return GestureState.SECTION_ACTIVE;

    if (hands.length >= 2) return GestureState.TWO_HAND_ZOOM;

    if (this.pinchActive[0]) {
      // Once a drag has actually escalated to ROTATING, keep reporting
      // ROTATING (not PINCHING) so the debouncer doesn't see it as churn.
      return this.confirmedState === GestureState.ROTATING ? GestureState.ROTATING : GestureState.PINCHING;
    }

    const isOpenPalm = hand0.extended.index && hand0.extended.middle && hand0.extended.ring && hand0.extended.pinky;
    if (isOpenPalm) return GestureState.PANNING;

    const isPointing = hand0.extended.index && !hand0.extended.middle && !hand0.extended.ring && !hand0.extended.pinky;
    if (isPointing) return GestureState.POINTING;

    return GestureState.HAND_VISIBLE;
  }

  private advanceDebouncer(rawCandidate: GestureState): void {
    if (this.rawCandidateStreak.candidate === rawCandidate) {
      this.rawCandidateStreak.count++;
    } else {
      this.rawCandidateStreak = { candidate: rawCandidate, count: 1 };
    }

    if (rawCandidate === this.confirmedState) {
      this.missingStreak = 0;
      return;
    }

    const isUtilityGesture = rawCandidate === GestureState.PAUSE_TOGGLE || rawCandidate === GestureState.RESET_VIEW;
    const confirmFrames = rawCandidate === GestureState.SECTION_ACTIVE
      ? SECTION_CONFIRM_FRAMES
      : isUtilityGesture ? UTILITY_CONFIRM_FRAMES : CONFIRM_FRAMES;
    if (this.rawCandidateStreak.count >= confirmFrames) {
      this.transitionTo(rawCandidate);
      return;
    }

    const cancelFrames = this.confirmedState === GestureState.SECTION_ACTIVE ? SECTION_CANCEL_FRAMES : CANCEL_FRAMES;
    this.missingStreak++;
    if (this.missingStreak >= cancelFrames && this.confirmedState !== GestureState.HAND_VISIBLE) {
      this.transitionTo(GestureState.HAND_VISIBLE);
    }
  }

  private transitionTo(next: GestureState): void {
    if (next === this.confirmedState) return;
    if (this.confirmedState === GestureState.SECTION_ACTIVE) this.emit('sectionEnd', undefined);

    this.confirmedState = next;
    this.missingStreak = 0;
    this.baseline = {};

    if (next === GestureState.SECTION_ACTIVE) this.emit('sectionStart', undefined);
    if (next === GestureState.PAUSE_TOGGLE) this.emit('pauseToggle', undefined);
    if (next === GestureState.RESET_VIEW) this.emit('resetView', undefined);
  }

  private driveActiveState(hands: HandFrame[]): void {
    const hand0 = hands[0];

    switch (this.confirmedState) {
      case GestureState.SECTION_ACTIVE:
        this.driveSection(hand0);
        break;
      case GestureState.TWO_HAND_ZOOM:
        this.driveZoom(hands);
        break;
      case GestureState.PINCHING:
      case GestureState.ROTATING:
        this.driveRotate(hand0);
        break;
      case GestureState.PANNING:
        this.drivePan(hand0);
        break;
      case GestureState.POINTING:
        this.emit('pointer', { x: hand0.indexTip.x, y: hand0.indexTip.y });
        break;
      default:
        break;
    }
  }

  private driveSection(hand0: HandFrame): void {
    const angle = handAngle(hand0.landmarks[5], hand0.landmarks[17]);
    if (this.baseline.handY === undefined || this.baseline.handAngle === undefined) {
      this.baseline.handY = hand0.center.y;
      this.baseline.handAngle = angle;
      return;
    }

    const delta = applyDeadZone(hand0.center.y - this.baseline.handY, DEAD_ZONE);
    this.baseline.handY = hand0.center.y;
    if (delta !== 0) this.emit('sectionMove', { delta });

    const deltaAngle = applyDeadZone(angleDelta(angle, this.baseline.handAngle), DEAD_ZONE);
    this.baseline.handAngle = angle;
    if (deltaAngle !== 0) this.emit('sectionRotate', { deltaAngle });
  }

  private driveZoom(hands: HandFrame[]): void {
    if (hands.length < 2) return;
    const dist = distance2D(hands[0].center, hands[1].center);
    if (this.baseline.twoHandDistance === undefined) {
      this.baseline.twoHandDistance = dist;
      return;
    }
    if (this.baseline.twoHandDistance > 1e-4) {
      this.emit('zoom', { scale: dist / this.baseline.twoHandDistance });
    }
    this.baseline.twoHandDistance = dist;
  }

  private driveRotate(hand0: HandFrame): void {
    if (this.baseline.handCenter === undefined) {
      this.baseline.handCenter = { ...hand0.center };
      return;
    }
    const dx = applyDeadZone(hand0.center.x - this.baseline.handCenter.x, DEAD_ZONE);
    const dy = applyDeadZone(hand0.center.y - this.baseline.handCenter.y, DEAD_ZONE);
    this.baseline.handCenter = { ...hand0.center };
    if (dx === 0 && dy === 0) return;

    // First real movement while pinching escalates PINCHING -> ROTATING.
    this.confirmedState = GestureState.ROTATING;
    this.emit('rotate', { dx, dy });
  }

  private drivePan(hand0: HandFrame): void {
    if (this.baseline.handCenter === undefined) {
      this.baseline.handCenter = { ...hand0.center };
      return;
    }
    const dx = applyDeadZone(hand0.center.x - this.baseline.handCenter.x, DEAD_ZONE);
    const dy = applyDeadZone(hand0.center.y - this.baseline.handCenter.y, DEAD_ZONE);
    this.baseline.handCenter = { ...hand0.center };
    if (dx !== 0 || dy !== 0) this.emit('pan', { dx, dy });
  }
}
