"""The rapid frame-labeler (Tier 2).

A small Gradio app: it shows you candidate frames one at a time and you rate
each "want it" / "skip" (or use the keyboard). Candidates are seeded by the
current scorer (Tier-1 similarity, or an existing model), so you're rating the
frames that actually matter — active learning, fewer labels for the same signal.

Verdicts are written straight into the LabelStore; run `peaks train` afterward.

Gradio lives in the `[label]` extra and ffmpeg must be on PATH (frames are
grabbed on demand from the original files, so nothing extra is stored). This
module is thin glue around the tested pieces (candidates, label store, grabber).
"""

from __future__ import annotations

from typing import Callable

from .labels import LabelStore
from .pipeline import Candidate


def launch_labeler(
    candidates: list[Candidate],
    store: LabelStore,
    profile: str,
    grab_frame: Callable[[str, float], "object"],
    *,
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Open the labeling UI over `candidates`, persisting verdicts to `store`."""
    import gradio as gr  # lazy (the `[label]` extra)

    state = {"i": 0}

    def _render(i: int):
        if i >= len(candidates):
            pos, neg = store.counts(profile)
            return None, f"Done — {len(candidates)} candidates reviewed. " \
                         f"Labels for '{profile}': {pos} positive / {neg} negative.", ""
        c = candidates[i]
        try:
            img = grab_frame(c.path, c.time)
        except Exception as exc:  # skip unreadable frames gracefully
            img = None
            note = f"(couldn't load frame: {exc})"
        else:
            note = ""
        caption = (
            f"[{i + 1}/{len(candidates)}] scene {c.scene_id} @ {c.time:.1f}s "
            f"· score {c.score:.3f} {note}"
        )
        return img, caption, ""

    def _record(label_value: int):
        i = state["i"]
        if i < len(candidates):
            c = candidates[i]
            store.add(c.key, c.time, label_value, profile, scene_id=c.scene_id)
            store.save()
            state["i"] = i + 1
        return _render(state["i"])

    # hotkeys: → / y = want it, ← / n = skip (rating hundreds of frames by
    # mouse is misery). The JS just clicks the buttons, so both paths share
    # the exact same handler.
    hotkeys_js = """
    () => {
      const press = (id) => {
        const el = document.getElementById(id);
        if (!el) return;
        // elem_id may land on the <button> itself or a wrapper around it
        (el.tagName === 'BUTTON' ? el : el.querySelector('button') || el).click();
      };
      document.addEventListener('keydown', (e) => {
        if (e.target && /input|textarea/i.test(e.target.tagName)) return;
        if (e.key === 'ArrowRight' || e.key === 'y') press('want-btn');
        if (e.key === 'ArrowLeft' || e.key === 'n') press('skip-btn');
      });
    }
    """

    with gr.Blocks(title=f"Opus labeler — {profile}", js=hotkeys_js) as demo:
        gr.Markdown(
            f"### Opus labeler — profile `{profile}`\n"
            "Rate each frame — **→ / y** = want it, **← / n** = skip."
        )
        image = gr.Image(label="candidate", height=480)
        caption = gr.Markdown()
        with gr.Row():
            skip_btn = gr.Button("✗ Skip (no)  [←]", elem_id="skip-btn")
            want_btn = gr.Button(
                "✓ Want it (yes)  [→]", variant="primary", elem_id="want-btn"
            )
        _ = gr.Markdown()  # spacer

        want_btn.click(lambda: _record(1), outputs=[image, caption, _])
        skip_btn.click(lambda: _record(0), outputs=[image, caption, _])
        demo.load(lambda: _render(state["i"]), outputs=[image, caption, _])

    demo.launch(server_port=server_port, share=share)
