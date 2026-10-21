// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Markers, as Filmora has them: a flag dropped at the playhead on the
//! ruler or on the selected clip, by the tray's button, the M key or the
//! clip's menu; clicked to go to, dragged to move, double-clicked for a
//! sheet with its name, colour and note, right-clicked for its menu.
//!
//! The marks themselves are the document's (`concat_project`'s
//! `AddMarker`, `UpdateMarker` and `RemoveMarker`), so every change is an
//! undo step and an API caller reads them in `project.get`. What lives
//! here is the window's side: which marker a menu or the sheet is open on,
//! and what the lanes are handed to draw.

use concat_project::Command;
use concat_project::model::{MARKER_COLORS, Marker};

use crate::format::{colour_of, frames_timecode};
use crate::i18n::{t, tf};
use crate::studio::Studio;
use crate::ui::{Glyph, MarkerData, MarkerSheetData, MenuItemData, MenuRow};

/// One marker, by where it is kept: on the ruler (`clip_id` None) or on a
/// clip. A split clip's pieces share their markers' ids, so the clip is
/// part of the name.
#[derive(Clone, PartialEq, Debug)]
pub struct MarkerRef {
    pub id: String,
    pub clip_id: Option<String>,
}

impl MarkerRef {
    /// From the lanes' two strings, where an empty clip id is the ruler.
    pub fn new(id: &str, clip_id: &str) -> MarkerRef {
        MarkerRef {
            id: id.to_owned(),
            clip_id: (!clip_id.is_empty()).then(|| clip_id.to_owned()),
        }
    }
}

/// How far past a clip's ends the playhead still counts as over it.
const EDGE: f64 = 1e-6;

impl Studio {
    /// The active timeline's markers as the lanes draw them: the ruler's,
    /// and each clip's that falls inside what the clip shows.
    pub(crate) fn marker_rows(&self) -> Vec<MarkerData> {
        let timeline = self.timeline();
        let row = |marker: &Marker, clip_id: &str, lane: i32, seconds: f64| MarkerData {
            id: marker.id.as_str().into(),
            clip_id: clip_id.into(),
            row: lane,
            seconds: seconds as f32,
            name: marker.name.as_str().into(),
            note: marker.note.as_str().into(),
            color: colour_of(&marker.color),
        };
        let mut rows: Vec<MarkerData> = timeline
            .markers
            .iter()
            .map(|marker| row(marker, "", -1, marker.at))
            .collect();
        for clip in &timeline.clips {
            let lane = self.row_of(&clip.track_id);
            for marker in &clip.markers {
                if let Some(seconds) = clip.marker_time(marker) {
                    rows.push(row(marker, &clip.id, lane, seconds));
                }
            }
        }
        rows
    }

    /// The selected clips the playhead is over, on lanes that are not
    /// locked: what a marker "on the clip" lands on.
    fn markable_clips(&self) -> Vec<String> {
        let at = f64::from(self.playhead);
        self.selection
            .iter()
            .filter_map(|id| self.clip(id))
            .filter(|clip| {
                clip.start - EDGE <= at
                    && at <= clip.start + clip.duration + EDGE
                    && !self.locked(&clip.track_id)
            })
            .map(|clip| clip.id.clone())
            .collect()
    }

    /// Whether the tray's "on the selected clip" row has a clip to mark.
    pub(crate) fn can_mark_clip(&self) -> bool {
        !self.markable_clips().is_empty()
    }

    /// A marker at the playhead. `target` is "clip" for the selected clips
    /// under it, "timeline" for the ruler, and anything else for the clips
    /// when there are any and the ruler when not - Filmora's rule.
    pub fn add_marker(&mut self, target: &str) {
        self.flush_commit();
        let time = f64::from(self.playhead);
        let clips = self.markable_clips();
        let on_clips = match target {
            "clip" => true,
            "timeline" => false,
            _ => !clips.is_empty(),
        };
        if on_clips && clips.is_empty() {
            self.notify(&t("marker.needsClip"), false);
            return;
        }
        let at = |clip_id: Option<String>| Command::AddMarker {
            clip_id,
            time,
            name: None,
            color: None,
            note: None,
        };
        let mut commands: Vec<Command> = if on_clips {
            clips.into_iter().map(|id| at(Some(id))).collect()
        } else {
            vec![at(None)]
        };
        let command = if commands.len() == 1 {
            commands.remove(0)
        } else {
            Command::Batch { commands }
        };
        self.apply(command);
    }

    /// A marker on clip `clip_id` at the playhead, from the clip's menu.
    pub(crate) fn mark_clip_here(&mut self, clip_id: &str) {
        self.apply(Command::AddMarker {
            clip_id: Some(clip_id.to_owned()),
            time: f64::from(self.playhead),
            name: None,
            color: None,
            note: None,
        });
    }

    /// A marker dragged on the lanes to `seconds`.
    pub fn move_marker(&mut self, marker: MarkerRef, seconds: f32) {
        self.apply(Command::UpdateMarker {
            clip_id: marker.clip_id,
            marker_id: marker.id,
            time: Some(f64::from(seconds)),
            name: None,
            color: None,
            note: None,
        });
    }

    /// The marker, where it sits on the timeline, and the name of the clip
    /// it rides, if it rides one.
    fn find_marker(&self, marker: &MarkerRef) -> Option<(Marker, f64, Option<String>)> {
        match &marker.clip_id {
            None => self
                .timeline()
                .markers
                .iter()
                .find(|kept| kept.id == marker.id)
                .map(|kept| (kept.clone(), kept.at, None)),
            Some(clip_id) => {
                let clip = self.clip(clip_id)?;
                let kept = clip.markers.iter().find(|kept| kept.id == marker.id)?;
                let at = clip.marker_time(kept).unwrap_or(clip.start);
                Some((kept.clone(), at, Some(clip.name.clone())))
            }
        }
    }

    /// Opens the sheet on a marker, from a double click or its menu.
    pub fn edit_marker(&mut self, marker: MarkerRef) {
        self.flush_commit();
        if self.find_marker(&marker).is_some() {
            self.marker_sheet = Some(marker);
        }
    }

    /// What the sheet shows: closed when no marker is open in it, or the
    /// one it was open on has gone (an undo took it).
    pub(crate) fn marker_sheet_data(&self) -> MarkerSheetData {
        let Some((marker, at, clip)) = self
            .marker_sheet
            .as_ref()
            .and_then(|open| self.find_marker(open))
        else {
            return MarkerSheetData::default();
        };
        let when = frames_timecode(at as f32, self.frame_rate());
        MarkerSheetData {
            open: true,
            place: match clip {
                None => tf("marker.onTimeline", &[&when]),
                Some(name) => tf("marker.onClip", &[&name, &when]),
            }
            .into(),
            name: marker.name.as_str().into(),
            note: marker.note.as_str().into(),
            color: MARKER_COLORS
                .iter()
                .position(|known| known.eq_ignore_ascii_case(&marker.color))
                .unwrap_or(0) as i32,
        }
    }

    /// The sheet's Save: name, note and colour as one step.
    pub fn save_marker(&mut self, name: &str, note: &str, color: i32) {
        let Some(marker) = self.marker_sheet.take() else {
            return;
        };
        let color = usize::try_from(color)
            .ok()
            .and_then(|index| MARKER_COLORS.get(index))
            .unwrap_or(&MARKER_COLORS[0]);
        self.apply(Command::UpdateMarker {
            clip_id: marker.clip_id,
            marker_id: marker.id,
            time: None,
            name: Some(name.to_owned()),
            color: Some((*color).to_owned()),
            note: Some(note.to_owned()),
        });
    }

    /// The sheet's Delete.
    pub fn delete_open_marker(&mut self) {
        if let Some(marker) = self.marker_sheet.take() {
            self.delete_marker(marker);
        }
    }

    fn delete_marker(&mut self, marker: MarkerRef) {
        self.apply(Command::RemoveMarker {
            clip_id: marker.clip_id,
            marker_id: marker.id,
        });
    }

    /// The right-click menu on a marker.
    pub(crate) fn marker_menu_rows(&self, marker: &MarkerRef) -> Vec<MenuItemData> {
        let row = |id: &str, label: String, glyph: Glyph, danger: bool| MenuItemData {
            id: id.into(),
            label: label.into(),
            kind: MenuRow::Action,
            glyph,
            shortcut: "".into(),
            enabled: true,
            danger,
            checkable: false,
            checked: false,
        };
        let mut rows = vec![
            row("marker-edit", t("marker.edit"), Glyph::Marker, false),
            row("marker-delete", t("marker.delete"), Glyph::Trash, true),
        ];
        if marker.clip_id.is_none() && self.timeline().markers.len() > 1 {
            rows.push(MenuItemData {
                kind: MenuRow::Separator,
                ..Default::default()
            });
            rows.push(row(
                "marker-delete-all",
                t("marker.deleteAll"),
                Glyph::Trash,
                true,
            ));
        }
        rows
    }

    /// One of the marker menu's rows, on the marker it was opened on.
    pub fn marker_action(&mut self, action: &str) {
        let Some(marker) = self.menu_marker.take() else {
            return;
        };
        match action {
            "marker-edit" => self.edit_marker(marker),
            "marker-delete" => self.delete_marker(marker),
            "marker-delete-all" => {
                let commands: Vec<Command> = self
                    .timeline()
                    .markers
                    .iter()
                    .map(|kept| Command::RemoveMarker {
                        clip_id: None,
                        marker_id: kept.id.clone(),
                    })
                    .collect();
                self.apply(Command::Batch { commands });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_clip_id_is_the_ruler() {
        assert_eq!(MarkerRef::new("k1", "").clip_id, None);
        assert_eq!(MarkerRef::new("k1", "c4").clip_id.as_deref(), Some("c4"));
    }

    #[test]
    fn the_sheet_swatches_are_the_models_colours() {
        let sheet = include_str!("../ui/dialogs/marker.slint");
        for color in MARKER_COLORS {
            assert!(sheet.contains(color), "{color} is not a swatch on the sheet");
        }
    }
}
