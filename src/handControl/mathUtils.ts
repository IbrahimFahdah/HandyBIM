import type { Landmark, Point2D } from './types';

export function distance2D(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Exponential smoothing: `alpha` is the weight given to the new sample. */
export function smooth(previous: number, current: number, alpha: number): number {
  return previous * (1 - alpha) + current * alpha;
}

export function applyDeadZone(value: number, threshold: number): number {
  return Math.abs(value) < threshold ? 0 : value;
}

// A finger reads as "extended" when its tip sits farther from the wrist than
// its PIP joint does — a simple, orientation-tolerant curl heuristic.
export function isFingerExtended(landmarks: Landmark[], tipIndex: number, pipIndex: number, wristIndex = 0): boolean {
  const wrist = landmarks[wristIndex];
  const tipDistance = distance2D(landmarks[tipIndex], wrist);
  const pipDistance = distance2D(landmarks[pipIndex], wrist);
  return tipDistance > pipDistance;
}

/** Hand orientation angle from the index-MCP -> pinky-MCP vector. */
export function handAngle(indexMcp: Point2D, pinkyMcp: Point2D): number {
  return Math.atan2(pinkyMcp.y - indexMcp.y, pinkyMcp.x - indexMcp.x);
}

/** Shortest signed angular difference, wraparound-safe. */
export function angleDelta(current: number, previous: number): number {
  let delta = current - previous;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  return delta;
}
