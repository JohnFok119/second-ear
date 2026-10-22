// The audio element is the clock; everything else reads its currentTime.
export class Timeline {
  constructor(audio, duration, fps) {
    this.audio = audio; this.duration = duration; this.fps = fps;
  }
  get t() { return this.audio.currentTime; }
  get playing() { return !this.audio.paused && !this.audio.ended; }
  frameIndex(nFrames) { return Math.max(0, Math.min(nFrames - 1, Math.floor(this.t * this.fps))); }
  seek(t) { this.audio.currentTime = Math.max(0, Math.min(this.duration, t)); }
  async play() {
    if (this.audio.ended || this.t >= this.duration - 0.05) this.seek(0);
    await this.audio.play();
  }
  pause() { this.audio.pause(); }
}
