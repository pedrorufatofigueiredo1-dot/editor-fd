// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The pen mask, as Filmora draws one: the Mask section's Pen starts a
//! drawing on the monitor, each click lays a corner joined to the last,
//! and a click back on the first closes the shape, which becomes the
//! selected clip's mask in one step. The right button takes the last
//! corner back, and on none leaves the drawing.
//!
//! The drawing is held here in fractions of the frame, as the overlay
//! reports clicks; closing it hands the corners to `MaskFrame::pen_from`,
//! which lays them in the picture as the clip is placed - moved, scaled,
//! turned - so the mask keeps what was drawn over it.

use concat_project::commands::ClipPatch;
use concat_project::{Command, model};

use crate::i18n::t;
use crate::studio::Studio;
use crate::ui::PenPointData;

/// A drawing under way.
#[derive(Clone, Debug, PartialEq)]
pub struct PenDraft {
    /// The clip it will mask.
    pub clip_id: String,
    /// The corners so far, in fractions of the frame, in click order.
    pub points: Vec<(f32, f32)>,
}

impl Studio {
    /// The Mask section's Pen: a drawing starts on the selected picture.
    pub(crate) fn start_pen(&mut self) {
        self.flush_commit();
        let Some(clip_id) = self.sole_selection() else {
            self.notify(&t("mask.penNeedsClip"), false);
            return;
        };
        let picture = self
            .clip(&clip_id)
            .is_some_and(|clip| clip.kind.is_visual() && !self.locked(&clip.track_id));
        if !picture {
            self.notify(&t("mask.penNeedsClip"), false);
            return;
        }
        self.pen = Some(PenDraft {
            clip_id,
            points: Vec::new(),
        });
    }

    /// A click on the frame while drawing: one more corner.
    pub(crate) fn pen_clicked(&mut self, x: f32, y: f32) {
        if let Some(draft) = self.pen.as_mut()
            && draft.points.len() < model::MAX_PEN_POINTS
        {
            draft.points.push((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
        }
    }

    /// The right button while drawing: the last corner back, and with none
    /// left, no drawing.
    pub(crate) fn pen_back(&mut self) {
        if let Some(draft) = self.pen.as_mut()
            && draft.points.pop().is_none()
        {
            self.pen = None;
        }
    }

    /// A click on the first corner: the drawing becomes the clip's mask.
    pub(crate) fn pen_closed(&mut self) {
        let Some(draft) = self.pen.take() else {
            return;
        };
        let Some(clip) = self.clip(&draft.clip_id).cloned() else {
            return;
        };
        // The mask's frame for this picture, from the mask it has - its
        // softness and invert carry over - or a fresh pen's.
        let mut placed = clip.clone();
        placed.mask = Some(
            clip.mask
                .unwrap_or_else(|| model::ClipMask::of(model::MaskShape::Pen)),
        );
        let Some(frame) = self.mask_frame(&placed) else {
            return;
        };
        let (width, height) = self.output_size();
        let pixels: Vec<(f64, f64)> = draft
            .points
            .iter()
            .map(|&(x, y)| {
                (
                    f64::from(x) * f64::from(width),
                    f64::from(y) * f64::from(height),
                )
            })
            .collect();
        let Some((mask, corners)) = frame.pen_from(&pixels) else {
            self.notify(&t("mask.penTooFew"), false);
            return;
        };
        self.apply(Command::UpdateClip {
            clip_id: clip.id,
            patch: ClipPatch {
                mask: Some(Some(mask)),
                mask_points: Some(corners),
                ..Default::default()
            },
        });
    }

    /// The drawing's corners, for the overlay's dots.
    pub(crate) fn pen_rows(&self) -> Vec<PenPointData> {
        self.pen
            .iter()
            .flat_map(|draft| draft.points.iter())
            .map(|&(x, y)| PenPointData { x, y })
            .collect()
    }

    /// The drawing's lines so far, on the overlay's thousand-unit box.
    pub(crate) fn pen_path(&self) -> String {
        let Some(draft) = &self.pen else {
            return String::new();
        };
        draft
            .points
            .iter()
            .enumerate()
            .map(|(index, (x, y))| {
                let verb = if index == 0 { "M" } else { "L" };
                format!("{verb} {:.1} {:.1}", x * 1000.0, y * 1000.0)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
