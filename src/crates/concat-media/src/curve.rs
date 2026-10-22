// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Sound that follows a speed curve.
//!
//! A constant rate is a filter FFmpeg already has: `atempo` holds the pitch,
//! `asetrate` lets it go. A rate that changes over the clip is not - those
//! filters take one number - and cutting the clip into pieces of constant
//! rate, as the mix once did, left a seam every tenth of a second: each
//! piece's `atempo` started cold, each was placed to the millisecond, and
//! the sound came out chopped.
//!
//! So a curved clip's sound is made here, in one pass against the curve's
//! own map: the source decoded once at its own speed, then read at the
//! source time the curve says each output instant shows - the same
//! `consumed` the picture is drawn from, so sound and picture cannot drift
//! apart. Without pitch held that read is the tape effect: a windowed-sinc
//! interpolator whose band narrows as the rate climbs, so a fast stretch
//! does not alias. With pitch held it is WSOLA: overlapping windows laid at
//! a fixed output hop, each taken from where the curve points, nudged by
//! up to [`TOLERANCE`] to the offset that best continues the window before
//! it, so the waveform runs on without a seam whatever the rate does.
//!
//! What comes out is a 32-bit float WAV at [`MIX_RATE`], read by the mix
//! and the playback cache like any other file at speed one; the clip's own
//! filters, gain and fades then apply to it as they do to any clip.

use std::io::Write;
use std::path::Path;

use concat_core::SpeedCurve;

use crate::audio::MIX_RATE;
use crate::error::{Error, Result};
use crate::samples::{AudioDecoder, AudioOptions, SampleFormat};

/// WSOLA's window, in samples: 43 ms at the mix rate, long enough to hold
/// a low voice's period several times over.
const WINDOW: usize = 2048;
/// The output hop between windows: a quarter window, so four overlap at
/// every instant and a nudge in one is a quarter of what is heard.
const HOP: usize = 512;
/// How far a window may be nudged from where the curve points, in
/// samples: 11 ms either way, a period of anything above 90 Hz.
const TOLERANCE: i64 = 512;
/// The interpolator's half-width, in source samples at rates up to one;
/// a faster rate widens it in proportion, which is what narrows the band.
const TAPS: f64 = 8.0;

/// One curved clip's sound, as [`render`] makes it.
#[derive(Clone, Copy, Debug)]
pub struct CurveSource<'a> {
    /// The media file.
    pub path: &'a Path,
    /// Which of its audio streams, by stream index; `None` is the first.
    pub stream: Option<usize>,
    /// Seconds into the file where the clip begins.
    pub source_start: f64,
    /// How long the clip runs on the timeline, in seconds.
    pub duration: f64,
    /// The clip's speed over its length.
    pub curve: &'a SpeedCurve,
    /// True holds the pitch while the rate changes; false is the tape
    /// effect.
    pub preserve_pitch: bool,
}

/// Writes `source`'s sound, following its curve, to `destination` as a
/// stereo float WAV exactly `duration` long.
pub fn render(source: &CurveSource, destination: &Path) -> Result<()> {
    let map = Map {
        curve: source.curve,
        duration: source.duration.max(0.0),
    };
    let frames = (map.duration * f64::from(MIX_RATE)).round() as usize;
    // What the curve consumes, and enough beyond it for the last windows
    // and taps to read what is really there rather than silence.
    let window = source.curve.mean() * map.duration + 0.5;
    let mut decoder = AudioDecoder::open(
        source.path,
        &AudioOptions {
            start: Some(source.source_start.max(0.0)),
            duration: Some(window),
            filters: Vec::new(),
            rate: MIX_RATE,
            channels: 2,
            format: SampleFormat::F32,
            stream: source.stream,
        },
    )?;
    let mut feed = Feed::new(|| decoder.next_f32());

    let failed = |source: std::io::Error| Error::Io {
        path: destination.to_path_buf(),
        source,
    };
    let mut file = std::io::BufWriter::new(std::fs::File::create(destination).map_err(failed)?);
    file.write_all(&wav_header(frames as u64)).map_err(failed)?;
    let mut written = 0usize;
    let mut out = |samples: &[f32]| -> Result<()> {
        written += samples.len() / 2;
        let bytes: Vec<u8> = samples.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        file.write_all(&bytes).map_err(failed)
    };
    if source.preserve_pitch {
        stretch(&mut feed, &map, frames, &mut out)?;
    } else {
        resample(&mut feed, &map, frames, &mut out)?;
    }
    debug_assert_eq!(written, frames, "the sound is the clip's length");
    file.flush().map_err(failed)
}

/// A curve over a clip, in samples: where in the source each output
/// sample is, and how fast the source is going there.
struct Map<'a> {
    curve: &'a SpeedCurve,
    duration: f64,
}

impl Map<'_> {
    /// The source position, in samples from the clip's in-point, that
    /// output sample `frame` shows. Past either end the end's speed holds,
    /// so a window reaching over an end reads on rather than stopping.
    fn source_at(&self, frame: f64) -> f64 {
        let rate = f64::from(MIX_RATE);
        let seconds = frame / rate;
        if self.duration <= 0.0 {
            return frame;
        }
        let x = seconds / self.duration;
        let source = if x <= 0.0 {
            seconds * self.curve.speed_at(0.0)
        } else if x >= 1.0 {
            self.curve.consumed(1.0) * self.duration
                + (seconds - self.duration) * self.curve.speed_at(1.0)
        } else {
            self.curve.consumed(x) * self.duration
        };
        source * rate
    }

    /// The speed at output sample `frame`.
    fn speed_at(&self, frame: f64) -> f64 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        self.curve
            .speed_at(frame / f64::from(MIX_RATE) / self.duration)
    }
}

/// The decoded source, read forward: interleaved stereo pulled in as the
/// map reaches for it and let go behind it, so a long clip never sits in
/// memory whole.
struct Feed<F> {
    pull: F,
    /// Interleaved stereo from sample `base` on.
    samples: Vec<f32>,
    base: i64,
    ended: bool,
}

impl<F: FnMut() -> Result<Option<Vec<f32>>>> Feed<F> {
    fn new(pull: F) -> Self {
        Self {
            pull,
            samples: Vec::new(),
            base: 0,
            ended: false,
        }
    }

    /// Pulls until sample `frame` is held or the source is over.
    fn reach(&mut self, frame: i64) -> Result<()> {
        while !self.ended && self.base + (self.samples.len() / 2) as i64 <= frame {
            match (self.pull)()? {
                Some(chunk) => self.samples.extend_from_slice(&chunk),
                None => self.ended = true,
            }
        }
        Ok(())
    }

    /// Lets go of what comes before sample `frame`, in large steps so the
    /// shift is rare.
    fn forget(&mut self, frame: i64) {
        let drop = frame - self.base;
        let held = (self.samples.len() / 2) as i64;
        if drop > 1 << 16 && drop <= held {
            self.samples.drain(..drop as usize * 2);
            self.base = frame;
        }
    }

    /// One stereo sample; silence before the in-point and past the end.
    fn at(&self, frame: i64) -> [f32; 2] {
        let index = frame - self.base;
        if index < 0 || index as usize * 2 + 1 >= self.samples.len() {
            return [0.0, 0.0];
        }
        let at = index as usize * 2;
        [self.samples[at], self.samples[at + 1]]
    }

    /// Both channels together, which is what WSOLA lines up.
    fn mono(&self, frame: i64) -> f32 {
        let [left, right] = self.at(frame);
        left + right
    }
}

/// How well the `length` samples from `candidate` continue the ones from
/// `natural`: their correlation, over the candidate's own level so a loud
/// stretch is not preferred for being loud. Every `stride`th sample.
fn likeness<F>(feed: &Feed<F>, candidate: i64, natural: i64, length: usize, stride: usize) -> f64
where
    F: FnMut() -> Result<Option<Vec<f32>>>,
{
    let mut dot = 0.0f64;
    let mut energy = 0.0f64;
    for offset in (0..length).step_by(stride) {
        let x = f64::from(feed.mono(candidate + offset as i64));
        dot += x * f64::from(feed.mono(natural + offset as i64));
        energy += x * x;
    }
    dot / (energy + 1e-9).sqrt()
}

/// The pitch-holding retime: WSOLA against the map, `frames` samples of
/// it handed to `out` in order.
fn stretch<F>(
    feed: &mut Feed<F>,
    map: &Map,
    frames: usize,
    out: &mut impl FnMut(&[f32]) -> Result<()>,
) -> Result<()>
where
    F: FnMut() -> Result<Option<Vec<f32>>>,
{
    let window: Vec<f32> = (0..WINDOW)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / WINDOW as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect();
    let half = (WINDOW / 2) as i64;
    let hop = HOP as i64;
    let overlap = WINDOW - HOP;
    let total = frames as i64;

    // The sum of the windows laid so far, and of their weights, from
    // output sample `base` on; dividing one by the other is the output,
    // flat wherever the windows overlap however many do.
    let mut sum: Vec<f32> = Vec::new();
    let mut weight: Vec<f32> = Vec::new();
    let mut base = -half;
    let mut previous: Option<i64> = None;

    let mut emit = |sum: &mut Vec<f32>,
                    weight: &mut Vec<f32>,
                    base: &mut i64,
                    until: i64|
     -> Result<()> {
        let count = (until - *base).clamp(0, weight.len() as i64) as usize;
        let mut ready: Vec<f32> = Vec::with_capacity(count * 2);
        for index in 0..count {
            let at = *base + index as i64;
            if at < 0 || at >= total {
                continue;
            }
            let share = weight[index];
            let scale = if share > 1e-3 { 1.0 / share } else { 0.0 };
            ready.push(sum[index * 2] * scale);
            ready.push(sum[index * 2 + 1] * scale);
        }
        sum.drain(..count * 2);
        weight.drain(..count);
        *base += count as i64;
        if ready.is_empty() {
            Ok(())
        } else {
            out(&ready)
        }
    };

    let mut step = 0i64;
    while step * hop - half < total {
        let center = step * hop;
        // Where the curve says this window's source starts.
        let target = map.source_at(center as f64).round() as i64 - half;
        feed.reach(target + TOLERANCE + WINDOW as i64)?;
        let start = match previous {
            None => target,
            Some(previous) => {
                // What would follow the last window had nothing changed:
                // the window to line this one up with.
                let natural = previous + hop;
                feed.reach(natural + WINDOW as i64)?;
                let mut best = (f64::MIN, target);
                let consider = |candidate: i64, stride: usize, best: &mut (f64, i64)| {
                    let score = likeness(feed, candidate, natural, overlap, stride);
                    if score > best.0 {
                        *best = (score, candidate);
                    }
                };
                // Coarse, every fourth offset over every fourth sample,
                // then every offset around the best of those.
                let mut coarse = (f64::MIN, target);
                let mut offset = -TOLERANCE;
                while offset <= TOLERANCE {
                    let score = likeness(feed, target + offset, natural, overlap, 4);
                    if score > coarse.0 {
                        coarse = (score, target + offset);
                    }
                    offset += 4;
                }
                for candidate in coarse.1 - 3..=coarse.1 + 3 {
                    if (candidate - target).abs() <= TOLERANCE {
                        consider(candidate, 1, &mut best);
                    }
                }
                // The seamless choice wherever the tolerance allows it and
                // nothing is clearly better: at a steady rate of one it is
                // the source itself, and a shift of a whole period scores
                // the same but for rounding.
                let seamless = (natural - target).abs() <= TOLERANCE
                    && likeness(feed, natural, natural, overlap, 1) >= best.0 - 1e-3 * best.0.abs();
                if seamless { natural } else { best.1 }
            }
        };
        previous = Some(start);

        // Lay the window, growing the sums to hold it.
        let end = center + half;
        let needed = (end - base).max(0) as usize;
        if weight.len() < needed {
            weight.resize(needed, 0.0);
            sum.resize(needed * 2, 0.0);
        }
        for (index, &shape) in window.iter().enumerate() {
            let at = (center - half + index as i64 - base) as usize;
            let [left, right] = feed.at(start + index as i64);
            sum[at * 2] += left * shape;
            sum[at * 2 + 1] += right * shape;
            weight[at] += shape;
        }

        // Everything before the next window's start is final.
        emit(&mut sum, &mut weight, &mut base, center + hop - half)?;
        feed.forget(start.min(target - TOLERANCE));
        step += 1;
    }
    emit(&mut sum, &mut weight, &mut base, total)?;
    // Whatever the windows did not reach - nothing, unless the clip is
    // shorter than one - is silence, so the length is always the clip's.
    if base < total {
        let missing = (total - base.max(0)) as usize;
        out(&vec![0.0; missing * 2])?;
    }
    Ok(())
}

/// The tape-effect retime: every output sample read off the source where
/// the map puts it, through a windowed sinc as wide as the rate needs.
fn resample<F>(
    feed: &mut Feed<F>,
    map: &Map,
    frames: usize,
    out: &mut impl FnMut(&[f32]) -> Result<()>,
) -> Result<()>
where
    F: FnMut() -> Result<Option<Vec<f32>>>,
{
    const CHUNK: usize = 4096;
    let mut ready: Vec<f32> = Vec::with_capacity(CHUNK * 2);
    for frame in 0..frames {
        let at = map.source_at(frame as f64);
        // Past a rate of one the source is squeezed, and what was below
        // its Nyquist is not below the output's: the kernel widens by the
        // rate, which lowers its cutoff by as much.
        let widen = map.speed_at(frame as f64).max(1.0);
        let reach = (TAPS * widen).ceil() as i64;
        let floor = at.floor() as i64;
        feed.reach(floor + reach + 1)?;
        let (mut left, mut right, mut total) = (0.0f64, 0.0f64, 0.0f64);
        for tap in floor - reach + 1..=floor + reach {
            let distance = at - tap as f64;
            let shape = sinc(distance / widen) * hann(distance / reach as f64);
            let [l, r] = feed.at(tap);
            left += f64::from(l) * shape;
            right += f64::from(r) * shape;
            total += shape;
        }
        let scale = if total.abs() > 1e-9 { 1.0 / total } else { 0.0 };
        ready.push((left * scale) as f32);
        ready.push((right * scale) as f32);
        if ready.len() >= CHUNK * 2 {
            out(&ready)?;
            ready.clear();
            // Kept back by the widest kernel too, should the rate climb.
            feed.forget(floor - reach - (TAPS * SpeedCurve::MAX_SPEED) as i64);
        }
    }
    if !ready.is_empty() {
        out(&ready)?;
    }
    Ok(())
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let at = std::f64::consts::PI * x;
        at.sin() / at
    }
}

/// A raised cosine over `-1..=1`, zero outside it.
fn hann(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        0.0
    } else {
        0.5 + 0.5 * (std::f64::consts::PI * x).cos()
    }
}

/// The 44 bytes in front of `frames` stereo float samples at the mix rate.
fn wav_header(frames: u64) -> Vec<u8> {
    let data = (frames * 8).min(u64::from(u32::MAX - 36)) as u32;
    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data).to_le_bytes());
    header.extend_from_slice(b"WAVEfmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    // IEEE float, two channels.
    header.extend_from_slice(&3u16.to_le_bytes());
    header.extend_from_slice(&2u16.to_le_bytes());
    header.extend_from_slice(&MIX_RATE.to_le_bytes());
    header.extend_from_slice(&(MIX_RATE * 8).to_le_bytes());
    header.extend_from_slice(&8u16.to_le_bytes());
    header.extend_from_slice(&32u16.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data.to_le_bytes());
    header
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = MIX_RATE as f64;

    /// A feed over samples already in memory, handed out in frames the
    /// size a decoder hands them.
    fn feed_of(samples: Vec<f32>) -> Feed<impl FnMut() -> Result<Option<Vec<f32>>>> {
        let mut chunks = samples
            .chunks(2048)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>()
            .into_iter();
        Feed::new(move || Ok(chunks.next()))
    }

    /// A stereo sine of `hz`, `seconds` long.
    fn tone(hz: f64, seconds: f64) -> Vec<f32> {
        let count = (seconds * RATE) as usize;
        (0..count)
            .flat_map(|index| {
                let value = (std::f64::consts::TAU * hz * index as f64 / RATE).sin() as f32 * 0.5;
                [value, value]
            })
            .collect()
    }

    fn run(curve: &SpeedCurve, duration: f64, pitch: bool, source: Vec<f32>) -> Vec<f32> {
        let map = Map { curve, duration };
        let frames = (duration * RATE).round() as usize;
        let mut feed = feed_of(source);
        let mut all = Vec::new();
        let mut out = |samples: &[f32]| -> Result<()> {
            all.extend_from_slice(samples);
            Ok(())
        };
        if pitch {
            stretch(&mut feed, &map, frames, &mut out).expect("stretches");
        } else {
            resample(&mut feed, &map, frames, &mut out).expect("resamples");
        }
        all
    }

    fn left(samples: &[f32]) -> Vec<f32> {
        samples.iter().step_by(2).copied().collect()
    }

    /// The loudness of a stretch, as RMS.
    fn level(samples: &[f32]) -> f32 {
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// The tone's frequency, from its upward zero crossings.
    fn pitch_of(samples: &[f32]) -> f64 {
        let crossings = samples
            .windows(2)
            .filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0)
            .count();
        crossings as f64 * RATE / samples.len() as f64
    }

    fn ramp() -> SpeedCurve {
        SpeedCurve::new(&[(0.0, 0.5), (0.5, 2.0), (1.0, 0.75)]).expect("a curve")
    }

    #[test]
    fn the_sound_is_exactly_the_clip_long_both_ways() {
        let curve = ramp();
        for pitch in [true, false] {
            let out = run(&curve, 1.37, pitch, tone(440.0, 3.0));
            assert_eq!(out.len() / 2, (1.37 * RATE).round() as usize, "pitch {pitch}");
        }
    }

    /// The bug this module exists for: the sound of a curved clip must not
    /// dip anywhere. A steady tone through a ramp comes out steady - no
    /// stretch of ten milliseconds anywhere falls far under the level the
    /// whole has, which is what a seam between pieces did every 100 ms.
    #[test]
    fn a_tone_through_a_ramp_never_drops_out() {
        let curve = ramp();
        for pitch in [true, false] {
            let out = left(&run(&curve, 2.0, pitch, tone(220.0, 4.0)));
            let whole = level(&out);
            let slice = (RATE * 0.01) as usize;
            for (index, piece) in out.chunks(slice).enumerate() {
                if piece.len() < slice {
                    continue;
                }
                let here = level(piece);
                assert!(
                    here > whole * 0.8,
                    "pitch {pitch}: {here} at {}ms against {whole}",
                    index * 10
                );
            }
        }
    }

    #[test]
    fn holding_the_pitch_keeps_a_tone_where_it_was_at_any_rate() {
        for speed in [0.5, 1.0, 2.0, 3.0] {
            let curve = SpeedCurve::new(&[(0.0, speed)]).expect("a curve");
            let out = left(&run(&curve, 1.0, true, tone(440.0, 4.0)));
            let heard = pitch_of(&out[4800..43200]);
            assert!((heard - 440.0).abs() < 8.0, "at {speed}x the tone is {heard} Hz");
        }
    }

    #[test]
    fn letting_the_pitch_go_shifts_it_by_the_rate() {
        for speed in [0.5, 2.0] {
            let curve = SpeedCurve::new(&[(0.0, speed)]).expect("a curve");
            let out = left(&run(&curve, 1.0, false, tone(440.0, 3.0)));
            let heard = pitch_of(&out[4800..43200]);
            assert!(
                (heard - 440.0 * speed).abs() < 8.0,
                "at {speed}x the tone is {heard} Hz"
            );
        }
    }

    /// A rate of one, as a curve, is the source untouched: WSOLA finds the
    /// seamless continuation at every window and adds nothing to it.
    #[test]
    fn a_flat_curve_at_one_is_the_source_itself() {
        let curve = SpeedCurve::new(&[(0.0, 1.0)]).expect("a curve");
        let source = tone(330.0, 2.0);
        for pitch in [true, false] {
            let out = run(&curve, 1.0, pitch, source.clone());
            let worst = out
                .iter()
                .zip(&source)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 2e-3, "pitch {pitch}: off by {worst}");
        }
    }

    /// Sound and picture agree: a click in the source comes out where the
    /// curve says the picture shows it.
    #[test]
    fn a_click_lands_where_the_curve_puts_it() {
        let curve = ramp();
        let duration = 2.0;
        let click_at = 1.3; // source seconds from the in-point
        let mut source = vec![0.0f32; (4.0 * RATE) as usize * 2];
        let index = (click_at * RATE) as usize;
        for offset in 0..48 {
            source[(index + offset) * 2] = 0.9;
            source[(index + offset) * 2 + 1] = 0.9;
        }
        // Where the curve has consumed the click's source time.
        let mut expected = 0.0;
        for step in 0..=20_000 {
            let x = f64::from(step) / 20_000.0;
            if curve.consumed(x) * duration >= click_at {
                expected = x * duration;
                break;
            }
        }
        for pitch in [true, false] {
            let out = left(&run(&curve, duration, pitch, source.clone()));
            let loudest = out
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map(|(at, _)| at as f64 / RATE)
                .expect("sound");
            assert!(
                (loudest - expected).abs() < 0.03,
                "pitch {pitch}: the click is at {loudest}s, the picture at {expected}s"
            );
        }
    }

    /// End to end, as the export makes it: a tone in a file, a clip of it
    /// on a ramp, mixed to AAC. The mix is the clip's length and its sound
    /// never drops out - the chopping a ramp used to come out with.
    #[test]
    fn a_curved_clip_mixes_to_a_seamless_sound() {
        use crate::audio::{AudioClip, mix_to_file};

        let dir = std::env::temp_dir();
        let source = dir.join("concat-curve-test-tone.wav");
        let output = dir.join("concat-curve-test-mix.m4a");
        let samples = tone(220.0, 8.0);
        let mut bytes = wav_header((samples.len() / 2) as u64);
        bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
        std::fs::write(&source, bytes).expect("writes the tone");

        for pitch in [true, false] {
            let clip = AudioClip {
                path: source.clone(),
                stream: None,
                start: 0.5,
                duration: 3.0,
                source_start: 1.0,
                speed: ramp().mean(),
                retime: Some(ramp()),
                preserve_pitch: pitch,
                volume: 1.0,
                volume_curve: concat_core::animate::Track::default(),
                fade_in: 0.0,
                fade_out: 0.0,
                filter_chain: String::new(),
            };
            mix_to_file(&[clip], 4.0, &output).expect("mixes");
            let mut decoder = AudioDecoder::open(
                &output,
                &AudioOptions {
                    format: SampleFormat::F32,
                    ..AudioOptions::default()
                },
            )
            .expect("opens the mix");
            let mixed = left(&decoder.collect_f32().expect("decodes"));
            assert!(
                (mixed.len() as f64 / RATE - 4.0).abs() < 0.05,
                "pitch {pitch}: the mix is {}s",
                mixed.len() as f64 / RATE
            );
            // Inside the clip, clear of its two ends.
            let inside = &mixed[(0.55 * RATE) as usize..(3.45 * RATE) as usize];
            let whole = level(inside);
            let slice = (RATE * 0.01) as usize;
            for (index, piece) in inside.chunks_exact(slice).enumerate() {
                let here = level(piece);
                assert!(
                    here > whole * 0.7,
                    "pitch {pitch}: {here} at {}ms into the clip against {whole}",
                    50 + index * 10
                );
            }
        }
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&output);
        let leftovers = std::fs::read_dir(&dir)
            .expect("reads the temp folder")
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".concat-curve-test-mix.concat-curve")
            })
            .count();
        assert_eq!(leftovers, 0, "the curve's sound is cleared away");
    }

    #[test]
    fn the_header_describes_float_stereo_at_the_mix_rate() {
        let header = wav_header(10);
        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(u16::from_le_bytes([header[20], header[21]]), 3);
        assert_eq!(
            u32::from_le_bytes([header[40], header[41], header[42], header[43]]),
            80
        );
    }
}
