// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Markers: pins on the ruler and on clips, with a name, a colour and a
//! note.
//!
//! One arm per command, exactly as [`super::apply`] routes them here.

use super::*;
use crate::model::{MARKER_COLORS, Marker};

/// Applies one of this module's commands. Any other is a routing error.
pub(super) fn apply(
    project: &mut Project,
    mint: &mut IdMint,
    command: Command,
) -> Result<Outcome, CommandError> {
    let timeline = project.active_mut();
    match command {
        Command::AddMarker {
            clip_id,
            time,
            name,
            color,
            note,
        } => {
            let time = time.max(0.0);
            let mut marker = Marker {
                id: String::new(),
                at: time,
                name: name.map(|name| name.trim().to_owned()).unwrap_or_default(),
                color: color
                    .filter(|color| is_color(color))
                    .unwrap_or_else(|| MARKER_COLORS[0].to_owned()),
                note: note.unwrap_or_default(),
            };
            let markers = match &clip_id {
                Some(clip_id) => match timeline.clip_mut(clip_id) {
                    Some(clip) => {
                        marker.at = clip.source_at(time);
                        &mut clip.markers
                    }
                    None => return Ok(Outcome::default()),
                },
                None => &mut timeline.markers,
            };
            marker.id = mint.next("k");
            let id = marker.id.clone();
            markers.push(marker);
            Ok(Outcome {
                created_id: Some(id),
                applied: true,
            })
        }

        Command::UpdateMarker {
            clip_id,
            marker_id,
            time,
            name,
            color,
            note,
        } => {
            let (markers, at) = match &clip_id {
                Some(clip_id) => match timeline.clip_mut(clip_id) {
                    Some(clip) => {
                        let at = time.map(|time| clip.source_at(time.max(0.0)));
                        (&mut clip.markers, at)
                    }
                    None => return Ok(Outcome::default()),
                },
                None => (&mut timeline.markers, time.map(|time| time.max(0.0))),
            };
            let Some(marker) = markers.iter_mut().find(|marker| marker.id == marker_id) else {
                return Ok(Outcome::default());
            };
            let mut applied = false;
            if let Some(at) = at {
                applied |= assign(&mut marker.at, at);
            }
            if let Some(name) = name {
                applied |= assign(&mut marker.name, name.trim().to_owned());
            }
            if let Some(color) = color.filter(|color| is_color(color)) {
                applied |= assign(&mut marker.color, color);
            }
            if let Some(note) = note {
                applied |= assign(&mut marker.note, note);
            }
            Ok(Outcome {
                created_id: None,
                applied,
            })
        }

        Command::RemoveMarker { clip_id, marker_id } => {
            let markers = match &clip_id {
                Some(clip_id) => match timeline.clip_mut(clip_id) {
                    Some(clip) => &mut clip.markers,
                    None => return Ok(Outcome::default()),
                },
                None => &mut timeline.markers,
            };
            let before = markers.len();
            markers.retain(|marker| marker.id != marker_id);
            Ok(Outcome {
                created_id: None,
                applied: markers.len() != before,
            })
        }

        _ => unreachable!("commands::apply routes only this module's commands here"),
    }
}

/// `#rrggbb`, the one spelling a marker's colour is stored in.
fn is_color(text: &str) -> bool {
    text.len() == 7
        && text.starts_with('#')
        && text[1..].chars().all(|digit| digit.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Editor;

    fn editor_with_clip() -> (Editor, String) {
        let mut editor = Editor::new();
        let view = editor
            .apply(Command::AddTextClip {
                track_id: None,
                above: false,
                start: 10.0,
                style: None,
                duration: Some(4.0),
                offset_y: None,
            })
            .expect("a title");
        (editor, view.created_id.expect("an id"))
    }

    #[test]
    fn a_ruler_marker_is_added_named_coloured_and_undone() {
        let (mut editor, _) = editor_with_clip();
        let id = editor
            .apply(Command::AddMarker {
                clip_id: None,
                time: 3.5,
                name: Some("  Intro ".to_owned()),
                color: Some("#ff3b30".to_owned()),
                note: Some("start here".to_owned()),
            })
            .expect("added")
            .created_id
            .expect("an id");
        let marker = &editor.project().active().markers[0];
        assert_eq!(marker.id, id);
        assert_eq!(marker.at, 3.5);
        assert_eq!(marker.name, "Intro");
        assert_eq!(marker.color, "#ff3b30");
        assert_eq!(marker.note, "start here");
        editor.undo();
        assert!(editor.project().active().markers.is_empty());
    }

    #[test]
    fn a_colour_that_is_not_one_falls_back_to_the_first() {
        let (mut editor, _) = editor_with_clip();
        editor
            .apply(Command::AddMarker {
                clip_id: None,
                time: -2.0,
                name: None,
                color: Some("red".to_owned()),
                note: None,
            })
            .expect("added");
        let marker = &editor.project().active().markers[0];
        assert_eq!(marker.color, MARKER_COLORS[0]);
        assert_eq!(marker.at, 0.0, "floored at the start");
    }

    #[test]
    fn a_clip_marker_travels_with_the_clip() {
        let (mut editor, clip) = editor_with_clip();
        editor
            .apply(Command::AddMarker {
                clip_id: Some(clip.clone()),
                time: 11.0,
                name: None,
                color: None,
                note: None,
            })
            .expect("added");
        let placed = editor.project().active().clip(&clip).expect("clip").clone();
        assert_eq!(placed.marker_time(&placed.markers[0]), Some(11.0));
        editor
            .apply(Command::MoveClips {
                moves: vec![ClipMove {
                    clip_id: clip.clone(),
                    start: 20.0,
                    track_id: placed.track_id.clone(),
                }],
            })
            .expect("moved");
        let moved = editor.project().active().clip(&clip).expect("clip").clone();
        assert_eq!(moved.marker_time(&moved.markers[0]), Some(21.0));
    }

    #[test]
    fn a_marker_is_changed_field_by_field_and_removed() {
        let (mut editor, _) = editor_with_clip();
        let id = editor
            .apply(Command::AddMarker {
                clip_id: None,
                time: 1.0,
                name: Some("a".to_owned()),
                color: None,
                note: None,
            })
            .expect("added")
            .created_id
            .expect("an id");
        let same = editor
            .apply(Command::UpdateMarker {
                clip_id: None,
                marker_id: id.clone(),
                time: None,
                name: Some("a".to_owned()),
                color: None,
                note: None,
            })
            .expect("tolerated");
        assert!(!same.applied, "nothing changed");
        editor
            .apply(Command::UpdateMarker {
                clip_id: None,
                marker_id: id.clone(),
                time: Some(2.0),
                name: None,
                color: Some("#34c759".to_owned()),
                note: Some("n".to_owned()),
            })
            .expect("changed");
        let marker = &editor.project().active().markers[0];
        assert_eq!((marker.at, marker.name.as_str()), (2.0, "a"));
        assert_eq!((marker.color.as_str(), marker.note.as_str()), ("#34c759", "n"));
        editor
            .apply(Command::RemoveMarker {
                clip_id: None,
                marker_id: id,
            })
            .expect("removed");
        assert!(editor.project().active().markers.is_empty());
    }

    #[test]
    fn markers_round_trip_through_the_document() {
        let (mut editor, clip) = editor_with_clip();
        for clip_id in [None, Some(clip.clone())] {
            editor
                .apply(Command::AddMarker {
                    clip_id,
                    time: 12.0,
                    name: Some("here".to_owned()),
                    color: None,
                    note: None,
                })
                .expect("added");
        }
        let settings = crate::DocumentSettings {
            name: "p".to_owned(),
            width: 1920,
            height: 1080,
            rate_num: 30,
            rate_den: 1,
        };
        let back = Editor::from_document(&editor.to_document(&settings)).expect("reads");
        assert_eq!(back.project().active().markers.len(), 1);
        let restored = back.project().active().clip(&clip).expect("clip");
        assert_eq!(restored.markers[0].name, "here");
    }
}
