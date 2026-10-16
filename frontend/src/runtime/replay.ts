export function stepIndex(index: number, delta: number, length: number) {
  return Math.max(0, Math.min(Math.max(0, length - 1), index + delta));
}
export const replayInterval = (speed: number) => 1000 / speed; // Presentation cadence; not measured engine elapsed time.
