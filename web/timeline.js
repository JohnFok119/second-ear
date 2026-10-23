// The audio element is the clock; everything else reads its currentTime.
export class Timeline {
  constructor(audio, duration, fps) {
    this.audio = audio; this.duration = duration; this.fps = fps; this.pending = null;
    audio.addEventListener('loadedmetadata', () => this.flush());
    audio.addEventListener('seeked', () => { this.pending = null; });
  }
  // A seek asked for before the audio has metadata is held in `pending`: it drives the visual
  // time at once and is applied to the audio element when `loadedmetadata` arrives.
  get t() {
    if (this.pending !== null && Math.abs(this.audio.currentTime - this.pending) < 0.1) this.pending = null;
    return this.pending !== null ? this.pending : this.audio.currentTime;
  }
  get playing() { return !this.audio.paused && !this.audio.ended; }
  frameIndex(nFrames) { return Math.max(0, Math.min(nFrames - 1, Math.floor(this.t * this.fps))); }
  seek(t) {
    this.pending = Math.max(0, Math.min(this.duration, t));
    if (this.audio.readyState >= 1) this.flush();
  }
  flush() {
    if (this.pending === null) return;
    try { this.audio.currentTime = this.pending; } catch (e) { /* stays pending */ }
    // `pending` is cleared by the `seeked` event (or once currentTime matches), so the visual
    // time is right even if the audio element is slow to move.
  }
  async play() {
    if (this.audio.ended || this.t >= this.duration - 0.05) this.seek(0);
    this.flush();
    this.pending = null; // from here the audio is the clock
    await this.audio.play();
  }
  pause() { this.audio.pause(); }
}
