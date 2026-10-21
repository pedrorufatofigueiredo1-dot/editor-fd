// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Marks for the AI: clips, and empty stretches of a lane, picked on the
//! timeline so a person talking to an assistant can say "put a title on 2"
//! or "cut 1 and 3". Each mark is numbered in the order it was made, and a
//! number is never given to another mark while the first still stands, nor
//! moved when another is taken off: "3" has to go on meaning what it meant
//! when it was said.
//!
//! The marks are the project's but not the document's. They live beside it,
//! in `ai-marks.json` in the project folder - not in `cache/`, which has a
//! button that empties it - so the document's format does not change, and
//! an edit is never undone by marking something.

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The file the marks are kept in, in the project folder.
pub const FILE: &str = "ai-marks.json";

/// Two times this close are one time: a range is matched by its ends, and
/// the ends come back through a float conversion or two.
const SAME: f64 = 1e-6;

/// What a mark points at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Target {
    /// A clip, by id. It goes when the clip does.
    #[serde(rename = "clip")]
    Clip {
        #[serde(rename = "clipId")]
        clip_id: String,
    },
    /// A stretch of one lane, in timeline seconds. It goes when the lane does.
    #[serde(rename = "range")]
    Range {
        #[serde(rename = "trackId")]
        track_id: String,
        start: f64,
        end: f64,
    },
}

impl Target {
    /// Whether `self` and `other` point at the same thing.
    fn same(&self, other: &Target) -> bool {
        match (self, other) {
            (Target::Clip { clip_id: a }, Target::Clip { clip_id: b }) => a == b,
            (
                Target::Range {
                    track_id: a,
                    start: a_start,
                    end: a_end,
                },
                Target::Range {
                    track_id: b,
                    start: b_start,
                    end: b_end,
                },
            ) => a == b && (a_start - b_start).abs() < SAME && (a_end - b_end).abs() < SAME,
            _ => false,
        }
    }
}

/// One mark: its number and what it points at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    pub number: u32,
    #[serde(flatten)]
    pub target: Target,
}

/// A project's marks, and the number the next one takes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Marks {
    /// The next mark's number. Only climbs while any mark stands, so a
    /// number taken off is not handed to the next thing marked; back to 1
    /// once there are none.
    next: u32,
    marks: Vec<Mark>,
}

impl Marks {
    pub fn is_empty(&self) -> bool {
        self.marks.is_empty()
    }

    /// The marks, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &Mark> {
        self.marks.iter()
    }

    /// The number on `target`, if it is marked.
    pub fn number_of(&self, target: &Target) -> Option<u32> {
        self.marks
            .iter()
            .find(|mark| mark.target.same(target))
            .map(|mark| mark.number)
    }

    /// The number on clip `id`, if it is marked.
    pub fn number_of_clip(&self, id: &str) -> Option<u32> {
        self.marks.iter().find_map(|mark| match &mark.target {
            Target::Clip { clip_id } if clip_id == id => Some(mark.number),
            _ => None,
        })
    }

    /// Marks `target` with the next number and returns it; something
    /// already marked keeps the number it has.
    pub fn mark(&mut self, target: Target) -> u32 {
        if let Some(number) = self.number_of(&target) {
            return number;
        }
        let number = self.next.max(1);
        self.next = number + 1;
        self.marks.push(Mark { number, target });
        number
    }

    /// Takes the mark off `target`. The others keep their numbers. True
    /// when there was one.
    pub fn unmark(&mut self, target: &Target) -> bool {
        let before = self.marks.len();
        self.marks.retain(|mark| !mark.target.same(target));
        self.settle();
        self.marks.len() != before
    }

    /// Takes every mark off.
    pub fn clear(&mut self) {
        self.marks.clear();
        self.settle();
    }

    /// Drops the marks on clips that are not in `clips` and on lanes that
    /// are not in `tracks`: a clip deleted takes its mark with it. True
    /// when any went.
    pub fn prune(&mut self, clips: &HashSet<&str>, tracks: &HashSet<&str>) -> bool {
        let before = self.marks.len();
        self.marks.retain(|mark| match &mark.target {
            Target::Clip { clip_id } => clips.contains(clip_id.as_str()),
            Target::Range { track_id, .. } => tracks.contains(track_id.as_str()),
        });
        self.settle();
        self.marks.len() != before
    }

    /// With nothing marked, numbering starts again at 1.
    fn settle(&mut self) {
        if self.marks.is_empty() {
            self.next = 1;
        }
    }

    /// The marks kept in project folder `dir`. None kept, or a file that
    /// cannot be read, is none: the marks are a convenience, and never a
    /// reason a project does not open.
    pub fn load(dir: &Path) -> Marks {
        let path = dir.join(FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("ai marks: could not read {}: {error}", path.display());
                }
                return Marks::default();
            }
        };
        match serde_json::from_str::<Marks>(&text) {
            Ok(mut marks) => {
                // A file edited by hand, or by an older build, still
                // numbers past every mark it holds.
                let highest = marks.marks.iter().map(|mark| mark.number).max();
                marks.next = marks.next.max(highest.map_or(1, |n| n + 1));
                marks
            }
            Err(error) => {
                log::warn!("ai marks: {} is not readable: {error}", path.display());
                Marks::default()
            }
        }
    }

    /// Keeps the marks in project folder `dir`, or removes the file when
    /// there are none. Written beside and moved over, so a crash mid-write
    /// leaves the last good file.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        let path = dir.join(FILE);
        if self.marks.is_empty() {
            return match std::fs::remove_file(&path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
                _ => Ok(()),
            };
        }
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let partial = dir.join(format!("{FILE}.partial"));
        std::fs::write(&partial, text)?;
        std::fs::rename(&partial, &path)
    }
}

#[cfg(test)]
mod tests {
    use super::{FILE, Marks, Target};
    use std::collections::HashSet;

    fn clip(id: &str) -> Target {
        Target::Clip {
            clip_id: id.to_owned(),
        }
    }

    fn range(track: &str, start: f64, end: f64) -> Target {
        Target::Range {
            track_id: track.to_owned(),
            start,
            end,
        }
    }

    /// Marks are numbered 1, 2, 3 in the order they are made, and marking
    /// a thing twice keeps its first number.
    #[test]
    fn marks_are_numbered_in_the_order_they_are_made() {
        let mut marks = Marks::default();
        assert_eq!(marks.mark(clip("c4")), 1);
        assert_eq!(marks.mark(range("t1", 2.0, 3.5)), 2);
        assert_eq!(marks.mark(clip("c1")), 3);
        assert_eq!(marks.mark(clip("c4")), 1);
        assert_eq!(marks.mark(range("t1", 2.0, 3.5 + 1e-9)), 2);
        assert_eq!(marks.iter().count(), 3);
        assert_eq!(marks.number_of_clip("c1"), Some(3));
        assert_eq!(marks.number_of_clip("c2"), None);
    }

    /// Taking a mark off leaves the others' numbers alone, and the next
    /// mark does not reuse the number taken off. With none left, the
    /// count starts over.
    #[test]
    fn unmarking_does_not_renumber() {
        let mut marks = Marks::default();
        marks.mark(clip("a"));
        marks.mark(clip("b"));
        marks.mark(clip("c"));
        assert!(marks.unmark(&clip("b")));
        assert!(!marks.unmark(&clip("b")));
        assert_eq!(marks.number_of_clip("a"), Some(1));
        assert_eq!(marks.number_of_clip("c"), Some(3));
        assert_eq!(marks.mark(clip("d")), 4);

        marks.clear();
        assert!(marks.is_empty());
        assert_eq!(marks.mark(clip("e")), 1);
        marks.unmark(&clip("e"));
        assert_eq!(marks.mark(clip("f")), 1);
    }

    /// A deleted clip's mark goes, and a range goes with its lane; the
    /// rest keep their numbers.
    #[test]
    fn marks_on_deleted_clips_and_lanes_go() {
        let mut marks = Marks::default();
        marks.mark(clip("c1"));
        marks.mark(clip("c2"));
        marks.mark(range("t1", 0.0, 1.0));
        marks.mark(range("t2", 0.0, 1.0));
        let clips: HashSet<&str> = ["c2"].into();
        let tracks: HashSet<&str> = ["t2"].into();
        assert!(marks.prune(&clips, &tracks));
        assert!(!marks.prune(&clips, &tracks));
        let left: Vec<u32> = marks.iter().map(|mark| mark.number).collect();
        assert_eq!(left, [2, 4]);
        assert_eq!(marks.mark(clip("c3")), 5);
    }

    /// The file round-trips, says what each mark is in words an API
    /// caller reads, and is removed when nothing is marked. A missing or
    /// broken file is no marks.
    #[test]
    fn the_file_is_written_and_read_back() {
        let dir = std::env::temp_dir().join(format!("concat-ai-marks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(Marks::load(&dir).is_empty());

        let mut marks = Marks::default();
        marks.mark(clip("c7"));
        marks.mark(range("t3", 1.25, 4.0));
        marks.mark(clip("c9"));
        marks.unmark(&clip("c9"));
        marks.save(&dir).unwrap();

        let text = std::fs::read_to_string(dir.join(FILE)).unwrap();
        assert!(text.contains(r#""kind": "clip""#), "{text}");
        assert!(text.contains(r#""clipId": "c7""#), "{text}");
        assert!(text.contains(r#""trackId": "t3""#), "{text}");

        let back = Marks::load(&dir);
        assert_eq!(back, marks);
        let mut back = back;
        assert_eq!(back.mark(clip("c10")), 4);

        // A number past the counter, as a hand edit might leave it.
        std::fs::write(
            dir.join(FILE),
            r#"{"marks": [{"number": 8, "kind": "clip", "clipId": "c1"}]}"#,
        )
        .unwrap();
        assert_eq!(Marks::load(&dir).mark(clip("c2")), 9);

        std::fs::write(dir.join(FILE), "not json").unwrap();
        assert!(Marks::load(&dir).is_empty());

        Marks::default().save(&dir).unwrap();
        assert!(!dir.join(FILE).exists());
        Marks::default().save(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
