// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! A shape that keeps part of a clip's picture: a clip's mask.
//!
//! A mask is drawn analytically, as a signed distance from the shape's
//! edge, so it costs no texture and is sharp at any size. The distance is
//! measured in units of the picture's shorter side, which is what makes a
//! circle round on any aspect and a feather the same softness on a 720p
//! export and a 4K one. Negative is inside; the coverage falls from one to
//! zero across a band the feather sets, centred on the edge, and an
//! inverted mask keeps the outside instead.
//!
//! This is the one definition of the arithmetic. The GPU compositor runs a
//! line-for-line copy of it in WGSL (`concat-render`'s `MASK_WGSL`), and its
//! tests hold the two together; the CPU reference calls [`ShapeMask::coverage`]
//! directly. A change here is a change there.

/// Which shape a mask keeps. The order is the shader's numbering, one up:
/// zero there is "no mask".
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum MaskShape {
    /// One side of a straight line through the centre: the part above it,
    /// turned with the mask.
    Linear,
    /// A band either side of a line through the centre, `height` thick.
    Mirror,
    /// An ellipse filling the mask's box; a circle when it is square.
    #[default]
    Circle,
    /// The mask's box, its corners rounded by `roundness`.
    Rectangle,
    /// A heart filling the box, point down.
    Heart,
    /// A five-pointed star, point up.
    Star,
    /// A shape drawn point by point, as Filmora's pen draws one: a polygon
    /// whose corners are the clip's own (`pen_matte`), laid in the mask's
    /// box. Not drawn analytically - it has more corners than a uniform
    /// carries - so it is outside [`MaskShape::ALL`], and the executor
    /// hands it to the compositor as a matte.
    Pen,
}

impl MaskShape {
    /// Every shape, in the order a picker lists them.
    pub const ALL: [MaskShape; 6] = [
        MaskShape::Linear,
        MaskShape::Mirror,
        MaskShape::Circle,
        MaskShape::Rectangle,
        MaskShape::Heart,
        MaskShape::Star,
    ];

    /// The name a document writes. Stable for ever.
    pub fn name(self) -> &'static str {
        match self {
            MaskShape::Linear => "linear",
            MaskShape::Mirror => "mirror",
            MaskShape::Circle => "circle",
            MaskShape::Rectangle => "rectangle",
            MaskShape::Heart => "heart",
            MaskShape::Star => "star",
            MaskShape::Pen => "pen",
        }
    }

    /// The shape a document names, if this build knows it.
    pub fn parse(name: &str) -> Option<MaskShape> {
        MaskShape::ALL
            .into_iter()
            .chain([MaskShape::Pen])
            .find(|shape| shape.name() == name.trim())
    }

    /// The shader's number for this shape: one up from its place in
    /// [`MaskShape::ALL`], zero being no mask at all.
    pub fn code(self) -> f32 {
        match self {
            MaskShape::Linear => 1.0,
            MaskShape::Mirror => 2.0,
            MaskShape::Circle => 3.0,
            MaskShape::Rectangle => 4.0,
            MaskShape::Heart => 5.0,
            MaskShape::Star => 6.0,
            // The shader knows no seventh shape and keeps everything; a pen
            // reaches the compositor as a matte instead (`pen_matte`).
            MaskShape::Pen => 7.0,
        }
    }

    /// Whether the box's width means anything: a line has no width, and a
    /// band only a thickness.
    pub fn has_width(self) -> bool {
        !matches!(self, MaskShape::Linear | MaskShape::Mirror)
    }

    /// Whether the box's height means anything: a line has none.
    pub fn has_height(self) -> bool {
        self != MaskShape::Linear
    }
}

/// A mask, at one instant: everything the compositor needs to weigh a
/// pixel of the picture by it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ShapeMask {
    /// What it keeps.
    pub shape: MaskShape,
    /// The centre's offset from the picture's centre, in fractions of the
    /// picture's width; positive is right.
    pub x: f64,
    /// The same down the picture's height; positive is down.
    pub y: f64,
    /// The box's width, in the picture's shorter side.
    pub width: f64,
    /// The box's height, in the picture's shorter side; a band's thickness.
    pub height: f64,
    /// Clockwise degrees about the centre.
    pub rotation: f64,
    /// How round a rectangle's corners are: zero square, one as round as
    /// the box's shorter half allows.
    pub roundness: f64,
    /// How soft the edge is, `0..=1`: the band the coverage falls across,
    /// as a fraction of half the picture's shorter side.
    pub feather: f64,
    /// Keeps the outside of the shape instead of the inside.
    pub invert: bool,
}

impl Default for ShapeMask {
    /// A circle in the middle, two thirds of the picture's shorter side
    /// across, with a hard edge.
    fn default() -> Self {
        ShapeMask {
            shape: MaskShape::Circle,
            x: 0.0,
            y: 0.0,
            width: 0.66,
            height: 0.66,
            rotation: 0.0,
            roundness: 0.0,
            feather: 0.0,
            invert: false,
        }
    }
}

/// The parts of a mask the shader takes, packed as it takes them: three
/// four-vectors, so a layer's mask rides along on its vertices.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MaskUniform {
    /// Shape code, invert (0 or 1), the feather band and the corner radius,
    /// the last two in short-side units.
    pub a: [f32; 4],
    /// The centre in fractions of the picture, from its centre, and the
    /// box's half width and half height in short-side units.
    pub b: [f32; 4],
    /// The cosine and sine of the turn, and the picture's width and height
    /// over its shorter side.
    pub c: [f32; 4],
}

impl MaskUniform {
    /// No mask: every pixel kept.
    pub const NONE: MaskUniform = MaskUniform {
        a: [0.0; 4],
        b: [0.0; 4],
        c: [1.0, 0.0, 1.0, 1.0],
    };
}

impl ShapeMask {
    /// What the shader takes for this mask over a picture `width` by
    /// `height`, with `pixel` the size of one output pixel in the picture's
    /// shorter side: the narrowest band an edge may have, so a hard edge is
    /// still smoothed over a pixel rather than stair-stepped.
    pub fn uniform(&self, width: f32, height: f32, pixel: f32) -> MaskUniform {
        let short = width.min(height).max(1.0);
        let finite = |value: f64, fallback: f64| {
            if value.is_finite() {
                value as f32
            } else {
                fallback as f32
            }
        };
        let half_w = (finite(self.width, 0.66) / 2.0).max(1e-4);
        let half_h = (finite(self.height, 0.66) / 2.0).max(1e-4);
        let band = (finite(self.feather, 0.0).clamp(0.0, 1.0) * 0.5).max(pixel.max(1e-6));
        let radius = finite(self.roundness, 0.0).clamp(0.0, 1.0) * half_w.min(half_h);
        let (sin, cos) = finite(self.rotation, 0.0).to_radians().sin_cos();
        MaskUniform {
            a: [
                self.shape.code(),
                if self.invert { 1.0 } else { 0.0 },
                band,
                radius,
            ],
            b: [finite(self.x, 0.0), finite(self.y, 0.0), half_w, half_h],
            c: [cos, sin, width / short, height / short],
        }
    }

    /// [`ShapeMask::uniform`] for a picture pulled by `stretch` along its
    /// own axes as it is drawn: the mask is measured on the picture as it
    /// is seen pulled, in the unpulled picture's shorter side, so a pull
    /// moves the mask with the picture but never bends its shape - a circle
    /// stays round, a drawing keeps the shape it was drawn in - and the
    /// overlay's outline, which is measured that way, is the edge kept.
    pub fn uniform_stretched(
        &self,
        width: f32,
        height: f32,
        stretch: (f32, f32),
        pixel: f32,
    ) -> MaskUniform {
        let mut uniform = self.uniform(width, height, pixel);
        let short = width.min(height).max(1.0);
        let pull = |value: f32| {
            if value.is_finite() {
                value.abs().max(1e-6)
            } else {
                1.0
            }
        };
        uniform.c[2] = width * pull(stretch.0) / short;
        uniform.c[3] = height * pull(stretch.1) / short;
        uniform
    }

    /// How much of the picture is kept at `u`, `v` - `0..1` across it as
    /// it is seen - over a picture `width` by `height`, with `pixel` as in
    /// [`ShapeMask::uniform`].
    pub fn coverage(&self, u: f32, v: f32, width: f32, height: f32, pixel: f32) -> f32 {
        coverage(&self.uniform(width, height, pixel), u, v)
    }
}

/// How much of the picture a packed mask keeps at `u`, `v`: the shader's
/// `mask_coverage`, line for line.
pub fn coverage(mask: &MaskUniform, u: f32, v: f32) -> f32 {
    let [code, invert, band, radius] = mask.a;
    if code < 0.5 {
        return 1.0;
    }
    let [x, y, half_w, half_h] = mask.b;
    let [cos, sin, aspect_w, aspect_h] = mask.c;
    // The pixel's offset from the mask's centre in short-side units, then
    // turned back into the mask's own frame (clockwise in y-down).
    let dx = (u - 0.5 - x) * aspect_w;
    let dy = (v - 0.5 - y) * aspect_h;
    let p = [dx * cos + dy * sin, -dx * sin + dy * cos];
    let d = distance(code, p, [half_w, half_h], radius);
    let kept = 1.0 - smoothstep(-band * 0.5, band * 0.5, d);
    if invert > 0.5 { 1.0 - kept } else { kept }
}

/// The signed distance from the shape's edge at `p`, in the mask's own
/// frame and short-side units: negative inside.
pub fn distance(code: f32, p: [f32; 2], half: [f32; 2], radius: f32) -> f32 {
    let [hx, hy] = half;
    match code.round() as i32 {
        1 => p[1],
        2 => p[1].abs() - hy,
        3 => ellipse(p, half),
        4 => rounded_box(p, half, radius),
        5 => {
            // The heart's own frame: y up, its point at the origin, its
            // lobes' tops at 1.1036 and its sides at ±0.6036. Fitted to the
            // box and measured back in the box's units.
            const SIDE: f32 = 0.603_553_4;
            const MID: f32 = 0.551_776_7;
            let q = [p[0] / hx * SIDE, MID - p[1] / hy * MID];
            heart(q) * (hx / SIDE).min(hy / MID)
        }
        6 => star(p[0] / hx, -p[1] / hy) * hx.min(hy),
        _ => -1.0,
    }
}

/// WGSL's `smoothstep`, for `edge0 < edge1`.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// WGSL's `sign`: zero at zero, which Rust's `signum` is not.
fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn length(p: [f32; 2]) -> f32 {
    (p[0] * p[0] + p[1] * p[1]).sqrt()
}

fn dot2(p: [f32; 2]) -> f32 {
    p[0] * p[0] + p[1] * p[1]
}

/// An ellipse with these half axes. Exact on a circle; on an ellipse a
/// close approximation (Quilez's), which is what a feather band needs -
/// the edge itself, where the distance is zero, is exact either way.
fn ellipse(p: [f32; 2], r: [f32; 2]) -> f32 {
    let k0 = length([p[0] / r[0], p[1] / r[1]]);
    let k1 = length([p[0] / (r[0] * r[0]), p[1] / (r[1] * r[1])]);
    if k1 <= 1e-9 {
        return -r[0].min(r[1]);
    }
    k0 * (k0 - 1.0) / k1
}

/// A box with these half sides and corners rounded by `radius`.
fn rounded_box(p: [f32; 2], half: [f32; 2], radius: f32) -> f32 {
    let q = [p[0].abs() - half[0] + radius, p[1].abs() - half[1] + radius];
    length([q[0].max(0.0), q[1].max(0.0)]) + q[0].max(q[1]).min(0.0) - radius
}

/// Quilez's heart: y up, the point at the origin.
fn heart(p: [f32; 2]) -> f32 {
    let x = p[0].abs();
    let y = p[1];
    if y + x > 1.0 {
        return length([x - 0.25, y - 0.75]) - std::f32::consts::SQRT_2 / 4.0;
    }
    let m = (x + y).max(0.0) * 0.5;
    dot2([x, y - 1.0]).min(dot2([x - m, y - m])).sqrt() * sign(x - y)
}

/// The signed distance from a closed polygon's edge at `p`: negative
/// inside, by the even-odd rule, so a shape that crosses itself keeps what
/// a hand drawing it would expect. Quilez's `sdPolygon`. Fewer than three
/// corners is no shape: everything is outside.
pub fn polygon_distance(p: [f32; 2], corners: &[[f32; 2]]) -> f32 {
    if corners.len() < 3 {
        return f32::MAX;
    }
    let first = corners[0];
    let mut d = dot2([p[0] - first[0], p[1] - first[1]]);
    let mut s = 1.0;
    let mut j = corners.len() - 1;
    for (i, &vi) in corners.iter().enumerate() {
        let vj = corners[j];
        let e = [vj[0] - vi[0], vj[1] - vi[1]];
        let w = [p[0] - vi[0], p[1] - vi[1]];
        let along = ((w[0] * e[0] + w[1] * e[1]) / dot2(e).max(1e-12)).clamp(0.0, 1.0);
        let b = [w[0] - e[0] * along, w[1] - e[1] * along];
        d = d.min(dot2(b));
        let up = p[1] >= vi[1];
        let below = p[1] < vj[1];
        let left = e[0] * w[1] > e[1] * w[0];
        if (up && below && left) || (!up && !below && !left) {
            s = -s;
        }
        j = i;
    }
    s * d.sqrt()
}

impl ShapeMask {
    /// How much of the picture a pen mask keeps at `u`, `v`, as
    /// [`ShapeMask::coverage`] reckons the other shapes: `corners` are the
    /// polygon's, each a fraction of the mask's box from its centre, `-0.5`
    /// to `0.5` across and down, so moving, sizing, turning and softening
    /// the box move, size, turn and soften the drawing.
    pub fn pen_coverage(
        &self,
        corners: &[[f32; 2]],
        u: f32,
        v: f32,
        width: f32,
        height: f32,
        pixel: f32,
    ) -> f32 {
        let uniform = self.uniform(width, height, pixel);
        let shape = PenShape::of(&uniform, corners);
        shape.coverage(&uniform, u, v)
    }
}

/// A pen's corners laid out in the box's short-side units once, for the
/// many pixels a matte asks about.
struct PenShape {
    corners: Vec<[f32; 2]>,
}

impl PenShape {
    fn of(uniform: &MaskUniform, corners: &[[f32; 2]]) -> PenShape {
        let [_, _, half_w, half_h] = uniform.b;
        PenShape {
            corners: corners
                .iter()
                .map(|c| [c[0] * 2.0 * half_w, c[1] * 2.0 * half_h])
                .collect(),
        }
    }

    /// `coverage`'s arithmetic with the polygon's distance in the shape's.
    fn coverage(&self, mask: &MaskUniform, u: f32, v: f32) -> f32 {
        let [_, invert, band, _] = mask.a;
        let [x, y, _, _] = mask.b;
        let [cos, sin, aspect_w, aspect_h] = mask.c;
        let dx = (u - 0.5 - x) * aspect_w;
        let dy = (v - 0.5 - y) * aspect_h;
        let p = [dx * cos + dy * sin, -dx * sin + dy * cos];
        let d = polygon_distance(p, &self.corners);
        let kept = 1.0 - smoothstep(-band * 0.5, band * 0.5, d);
        if invert > 0.5 { 1.0 - kept } else { kept }
    }
}

/// The longest side of the matte a pen mask is drawn into: sampled across
/// the picture with filtering, so a few hundred texels make an edge as
/// clean as the screen shows, at a cost paid once per change of the mask.
pub const PEN_MATTE_SIDE: u32 = 768;

/// A pen mask drawn into a matte for a picture `width` by `height`: white,
/// its alpha what is kept, at most [`PEN_MATTE_SIDE`] on its longer side
/// and in the picture's aspect, which is what the compositor samples it
/// across. `None` for a picture with no size.
pub fn pen_matte(
    mask: &ShapeMask,
    corners: &[[f32; 2]],
    width: u32,
    height: u32,
    stretch: (f32, f32),
) -> Option<crate::frame::Frame> {
    matte_of(mask, Some(corners), width, height, stretch)
}

/// Any mask drawn into a matte over a picture - or a whole frame - `width`
/// by `height`, as [`pen_matte`] draws a pen's: the shapes by their own
/// distance, a pen by its corners. What a mask layer cuts the stack
/// beneath it with, the frame being its picture. `None` for no size, or
/// a pen with no corners.
pub fn matte_of(
    mask: &ShapeMask,
    corners: Option<&[[f32; 2]]>,
    width: u32,
    height: u32,
    stretch: (f32, f32),
) -> Option<crate::frame::Frame> {
    if width == 0 || height == 0 {
        return None;
    }
    let long = width.max(height) as f32;
    let fit = (PEN_MATTE_SIDE as f32 / long).min(1.0);
    let mw = ((width as f32 * fit).round() as u32).max(1);
    let mh = ((height as f32 * fit).round() as u32).max(1);
    // One matte texel in the picture's shorter side: the narrowest edge.
    let pixel = 1.0 / mw.min(mh) as f32;
    let uniform = mask.uniform_stretched(width as f32, height as f32, stretch, pixel);
    let pen = match (mask.shape, corners) {
        (MaskShape::Pen, Some(corners)) => Some(PenShape::of(&uniform, corners)),
        (MaskShape::Pen, None) => return None,
        _ => None,
    };
    let mut pixels = vec![255u8; (mw * mh * 4) as usize];
    for row in 0..mh {
        let v = (row as f32 + 0.5) / mh as f32;
        for column in 0..mw {
            let u = (column as f32 + 0.5) / mw as f32;
            let kept = match &pen {
                Some(shape) => shape.coverage(&uniform, u, v),
                None => coverage(&uniform, u, v),
            }
            .clamp(0.0, 1.0);
            pixels[((row * mw + column) * 4 + 3) as usize] = (kept * 255.0).round() as u8;
        }
    }
    crate::frame::Frame::from_rgba(mw, mh, pixels)
}

/// Quilez's five-pointed star, outer radius one, y up, a point straight up.
fn star(x: f32, y: f32) -> f32 {
    const K1: [f32; 2] = [0.809_017, -0.587_785_24];
    const K2: [f32; 2] = [-0.809_017, -0.587_785_24];
    const INNER: f32 = 0.45;
    let mut p = [x.abs(), y];
    let d1 = (K1[0] * p[0] + K1[1] * p[1]).max(0.0) * 2.0;
    p = [p[0] - d1 * K1[0], p[1] - d1 * K1[1]];
    let d2 = (K2[0] * p[0] + K2[1] * p[1]).max(0.0) * 2.0;
    p = [p[0] - d2 * K2[0], p[1] - d2 * K2[1]];
    p[0] = p[0].abs();
    p[1] -= 1.0;
    let ba = [INNER * -K1[1] - 0.0, INNER * K1[0] - 1.0];
    let h = ((p[0] * ba[0] + p[1] * ba[1]) / dot2(ba)).clamp(0.0, 1.0);
    length([p[0] - ba[0] * h, p[1] - ba[1] * h]) * sign(p[1] * ba[0] - p[0] * ba[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(shape: MaskShape) -> ShapeMask {
        ShapeMask {
            shape,
            ..ShapeMask::default()
        }
    }

    /// On a square picture of 100 pixels a side, one pixel is a hundredth.
    const PIXEL: f32 = 0.01;

    fn kept(mask: &ShapeMask, u: f32, v: f32) -> f32 {
        mask.coverage(u, v, 100.0, 100.0, PIXEL)
    }

    #[test]
    fn every_shape_keeps_its_middle_and_drops_its_far_corner() {
        for shape in MaskShape::ALL {
            let mask = ShapeMask {
                shape,
                width: 0.5,
                height: 0.5,
                ..ShapeMask::default()
            };
            // The line keeps the half above it, so its "middle" is a hair
            // above the centre.
            let (u, v) = if shape == MaskShape::Linear {
                (0.5, 0.3)
            } else {
                (0.5, 0.5)
            };
            assert!(kept(&mask, u, v) > 0.999, "{shape:?} keeps its middle");
            assert!(
                kept(&mask, 0.98, 0.98) < 0.001,
                "{shape:?} drops the corner"
            );
            let inverted = ShapeMask {
                invert: true,
                ..mask
            };
            assert!(kept(&inverted, u, v) < 0.001, "{shape:?} inverted");
            assert!(kept(&inverted, 0.98, 0.98) > 0.999, "{shape:?} inverted");
        }
    }

    #[test]
    fn a_circle_is_exact_and_round_on_a_wide_picture() {
        let circle = mask(MaskShape::Circle);
        // Radius 0.33 of the short side. On the edge, half kept.
        let d = distance(3.0, [0.33, 0.0], [0.33, 0.33], 0.0);
        assert!(d.abs() < 1e-6);
        let d = distance(3.0, [0.0, 0.5], [0.33, 0.33], 0.0);
        assert!((d - 0.17).abs() < 1e-6, "{d}");
        // On a 200 by 100 picture the short side is the height, so the
        // edge is 33 pixels from the centre both ways: u is 0.5 + 33/200.
        let wide = |u: f32, v: f32| circle.coverage(u, v, 200.0, 100.0, PIXEL);
        assert!((wide(0.5 + 0.33 / 2.0, 0.5) - 0.5).abs() < 1e-3);
        assert!((wide(0.5, 0.5 + 0.33) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn the_feather_is_a_smoothstep_centred_on_the_edge() {
        // Feather 0.4 is a band 0.2 of the short side wide, 0.1 each way.
        let soft = ShapeMask {
            feather: 0.4,
            ..mask(MaskShape::Circle)
        };
        let at = |r: f32| kept(&soft, 0.5 + r, 0.5);
        assert!((at(0.33) - 0.5).abs() < 1e-4, "half on the edge");
        assert!(at(0.33 - 0.1) > 0.9999, "all of it a band inside");
        assert!(at(0.33 + 0.1) < 1e-4, "none of it a band outside");
        // Halfway into the band, smoothstep's quarter: 0.15625 kept out.
        assert!((at(0.33 - 0.05) - 0.843_75).abs() < 1e-3);
        assert!((at(0.33 + 0.05) - 0.156_25).abs() < 1e-3);
        // A hard edge is still a pixel wide, not a step.
        let hard = mask(MaskShape::Circle);
        assert!((kept(&hard, 0.5 + 0.33, 0.5) - 0.5).abs() < 1e-3);
        assert!(kept(&hard, 0.5 + 0.33 + PIXEL, 0.5) < 1e-3);
    }

    #[test]
    fn a_rectangle_rounds_its_corners() {
        let half = [0.3, 0.2];
        // Square corners: the corner point is on the edge.
        assert!(distance(4.0, [0.3, 0.2], half, 0.0).abs() < 1e-6);
        // Fully round: the corner is cut, by r(√2 - 1) along the diagonal.
        let r = 0.2;
        let d = distance(4.0, [0.3, 0.2], half, r);
        assert!(
            (d - r * (std::f32::consts::SQRT_2 - 1.0)).abs() < 1e-5,
            "{d}"
        );
        // The middle of a side is untouched by the rounding.
        assert!(distance(4.0, [0.3, 0.0], half, r).abs() < 1e-6);
        assert!((distance(4.0, [0.0, 0.0], half, r) + 0.2).abs() < 1e-6);
    }

    #[test]
    fn the_line_and_the_band_turn_with_the_mask() {
        let line = mask(MaskShape::Linear);
        assert!(kept(&line, 0.5, 0.4) > 0.999);
        assert!(kept(&line, 0.5, 0.6) < 0.001);
        // Turned a quarter clockwise, "above" is the right: the line's
        // up points along +x once the frame is turned.
        let turned = ShapeMask {
            rotation: 90.0,
            ..line
        };
        assert!(kept(&turned, 0.6, 0.5) > 0.999);
        assert!(kept(&turned, 0.4, 0.5) < 0.001);
        let band = ShapeMask {
            height: 0.2,
            ..mask(MaskShape::Mirror)
        };
        assert!(kept(&band, 0.1, 0.5) > 0.999, "a band runs edge to edge");
        assert!(kept(&band, 0.5, 0.65) < 0.001);
        assert!((kept(&band, 0.5, 0.6) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn the_heart_and_the_star_fill_their_box() {
        let half = [0.3, 0.3];
        // The heart's point is on the box's bottom edge, its lobes at the
        // top, its sides at the box's sides.
        assert!(distance(5.0, [0.0, 0.3], half, 0.0).abs() < 1e-4);
        assert!(distance(5.0, [0.0, 0.0], half, 0.0) < 0.0);
        assert!(distance(5.0, [0.0, -0.31], half, 0.0) > 0.0);
        // The notch between the lobes is outside.
        assert!(distance(5.0, [0.0, -0.28], half, 0.0) > 0.0);
        // Its widest, where the lobes' circles are, is the box's side.
        assert!(distance(5.0, [0.3, -0.1077], half, 0.0).abs() < 1e-3);
        // The star's top point is on the box's top edge; between two
        // points it is outside, and its middle is in.
        assert!(distance(6.0, [0.0, -0.3], half, 0.0).abs() < 1e-4);
        assert!(distance(6.0, [0.0, -0.2], half, 0.0) < 0.0);
        assert!(distance(6.0, [0.0, 0.0], half, 0.0) < 0.0);
        let between = [
            0.3 * 36f32.to_radians().sin(),
            -0.3 * 36f32.to_radians().cos(),
        ];
        assert!(distance(6.0, between, half, 0.0) > 0.0);
    }

    #[test]
    fn a_moved_mask_is_measured_from_its_own_centre() {
        let moved = ShapeMask {
            x: 0.25,
            y: -0.25,
            width: 0.2,
            height: 0.2,
            ..mask(MaskShape::Rectangle)
        };
        assert!(kept(&moved, 0.75, 0.25) > 0.999);
        assert!(kept(&moved, 0.5, 0.5) < 0.001);
    }

    #[test]
    fn names_round_trip_and_nonsense_is_survivable() {
        for shape in MaskShape::ALL {
            assert_eq!(MaskShape::parse(shape.name()), Some(shape));
        }
        assert_eq!(MaskShape::parse("blob"), None);
        let broken = ShapeMask {
            width: f64::NAN,
            feather: f64::INFINITY,
            ..ShapeMask::default()
        };
        let value = kept(&broken, 0.5, 0.5);
        assert!(value.is_finite());
        assert_eq!(coverage(&MaskUniform::NONE, 0.1, 0.9), 1.0);
    }
}

#[cfg(test)]
mod pen_tests {
    use super::*;

    /// A pen mask whose box is the whole of a square picture.
    fn pen() -> ShapeMask {
        ShapeMask {
            shape: MaskShape::Pen,
            width: 1.0,
            height: 1.0,
            ..ShapeMask::default()
        }
    }

    /// The right half of the box, drawn as four corners.
    const RIGHT_HALF: [[f32; 2]; 4] = [[0.0, -0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 0.5]];

    #[test]
    fn a_polygon_is_negative_inside_and_exact_on_its_edge() {
        let square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        assert!((polygon_distance([0.0, 0.0], &square) + 1.0).abs() < 1e-6);
        assert!(polygon_distance([1.0, 0.0], &square).abs() < 1e-6);
        assert!((polygon_distance([3.0, 0.0], &square) - 2.0).abs() < 1e-6);
        // Wound the other way, it is the same shape.
        let mut back = square;
        back.reverse();
        assert!((polygon_distance([0.0, 0.0], &back) + 1.0).abs() < 1e-6);
        // Two corners are a line, not a shape.
        assert!(polygon_distance([0.0, 0.0], &square[..2]) > 1.0);
    }

    #[test]
    fn a_pen_keeps_what_was_drawn_and_turns_with_its_box() {
        let mask = pen();
        let kept = |m: &ShapeMask, u, v| m.pen_coverage(&RIGHT_HALF, u, v, 100.0, 100.0, 0.01);
        assert!(kept(&mask, 0.75, 0.5) > 0.999, "inside the drawing");
        assert!(kept(&mask, 0.25, 0.5) < 0.001, "outside it");
        let inverted = ShapeMask {
            invert: true,
            ..mask
        };
        assert!(kept(&inverted, 0.25, 0.5) > 0.999);
        // A half turn of the box puts the drawing on the left.
        let turned = ShapeMask {
            rotation: 180.0,
            ..mask
        };
        assert!(kept(&turned, 0.25, 0.5) > 0.999);
        assert!(kept(&turned, 0.75, 0.5) < 0.001);
    }

    #[test]
    fn a_pen_matte_is_in_the_pictures_aspect_and_keeps_by_its_alpha() {
        let matte = pen_matte(&pen(), &RIGHT_HALF, 1920, 1080, (1.0, 1.0)).expect("a matte");
        assert_eq!((matte.width(), matte.height()), (768, 432));
        let alpha = |x: u32, y: u32| matte.pixels()[((y * 768 + x) * 4 + 3) as usize];
        // The box is the picture's height square in the middle of a wide
        // picture: its right half is the middle of the frame and a bit.
        assert_eq!(alpha(500, 216), 255);
        assert_eq!(alpha(300, 216), 0);
        assert_eq!(alpha(700, 216), 0, "past the box");
        assert!(pen_matte(&pen(), &RIGHT_HALF, 0, 10, (1.0, 1.0)).is_none());
    }

    /// A shape's matte over a whole frame keeps the shape: a circle in the
    /// middle of a wide frame, round, and dark at the corners.
    #[test]
    fn a_shape_matte_covers_a_frame() {
        let circle = ShapeMask {
            shape: MaskShape::Circle,
            ..ShapeMask::default()
        };
        let matte = matte_of(&circle, None, 1920, 1080, (1.0, 1.0)).expect("a matte");
        assert_eq!((matte.width(), matte.height()), (768, 432));
        let alpha = |x: u32, y: u32| matte.pixels()[((y * 768 + x) * 4 + 3) as usize];
        assert_eq!(alpha(384, 216), 255, "the middle");
        assert_eq!(alpha(5, 5), 0, "a corner");
        // Radius 0.33 of the 432 short side is 142.6 texels, both ways.
        assert_eq!(alpha(384 + 135, 216), 255);
        assert_eq!(alpha(384 + 150, 216), 0);
        assert_eq!(alpha(384, 216 + 135), 255);
        assert!(matte_of(&pen(), None, 100, 100, (1.0, 1.0)).is_none());
    }

    /// A picture pulled to twice its width keeps a round circle: the edge
    /// is as far from the centre across as down, in the pixels drawn.
    #[test]
    fn a_pulled_picture_does_not_bend_its_mask() {
        let circle = ShapeMask {
            shape: MaskShape::Circle,
            ..ShapeMask::default()
        };
        let uniform = circle.uniform_stretched(100.0, 100.0, (2.0, 1.0), 0.01);
        // Radius 0.33 of the unpulled short side: 33 of the 200 pixels
        // drawn across is u 0.5 + 0.165; 33 of the 100 down is v 0.83.
        assert!((coverage(&uniform, 0.5 + 0.165, 0.5) - 0.5).abs() < 1e-3);
        assert!((coverage(&uniform, 0.5, 0.5 + 0.33) - 0.5).abs() < 1e-3);
        // Unpulled it is the uniform it always was.
        assert_eq!(
            circle.uniform_stretched(100.0, 100.0, (1.0, 1.0), 0.01),
            circle.uniform(100.0, 100.0, 0.01)
        );
    }
}
