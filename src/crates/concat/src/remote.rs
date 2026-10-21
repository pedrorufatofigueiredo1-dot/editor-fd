// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The window's side of the API it serves on the Remote page: a caller's
//! edits to the project on screen run here, on the event-loop thread,
//! through the same bookkeeping as an edit made by hand - the undo
//! history, the redraw, the autosave - so the person at the window sees
//! each one land and can undo it.

use std::sync::mpsc;
use std::time::Duration;

use concat_api::{WindowStatus, WindowWork};

use crate::host::Shell;

/// How long a call waits for the window. A modal dialog or a long stall
/// holds the event loop; the caller is told rather than kept waiting.
const PATIENCE: Duration = Duration::from_secs(20);

/// The window, as the API sees it.
pub struct Remote;

impl concat_api::Window for Remote {
    fn edit(&self, path: &str, work: WindowWork) -> bool {
        let path = path.to_owned();
        on_window(move || {
            let mut took = false;
            Shell::with(|shell, app| {
                took = shell.studio.borrow_mut().remote_edit(&path, work);
                shell.studio.borrow().publish(&app, &shell.models);
            });
            took
        })
        .unwrap_or(false)
    }

    fn status(&self) -> Option<WindowStatus> {
        on_window(|| {
            let mut status = None;
            Shell::with(|shell, _| status = shell.studio.borrow().remote_status());
            status
        })
        .flatten()
    }
}

/// Runs `body` on the event-loop thread and waits for what it returns.
fn on_window<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> Option<R> {
    let (sender, receiver) = mpsc::channel();
    slint::invoke_from_event_loop(move || {
        let _ = sender.send(body());
    })
    .ok()?;
    receiver.recv_timeout(PATIENCE).ok()
}
