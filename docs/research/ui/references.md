# UI references for the second-ear site (2026-11-15)

All sites below were opened live in Chrome on 2026-11-15, in a window 980 px wide. Nothing was clicked except
scrolling; cookie banners were left alone. The site list comes from `personal\private\notes\web-design-references.md`
and a reply from the osrs-website session (which quoted that note and its own `tasks\website\PLAN.md`).

## Johnny's taste, as already written down

- Chosen look for the Flip Assistant site (2026-11-13): **minta.framer.ai**. His words: "very clean", theme flavour
  "only slight".
- Pattern across that work (osrs-website session's reading): one clean dark page, one strong light idea, the product
  first, theme kept slight.
- Rejected sketches, as a taste signal: Terminal, Arcade, Old School, Studio.
- House rules (`design-kit` skill): no design skill by default; 14 px type floor; no sideways scroll at 390 px; the
  static page is approved before any animation.
- Johnny picks the look himself. This file lists what the references do and which patterns fit; it does not pick.

## What each site does

| Site | What is on screen | What second-ear could take from it |
|---|---|---|
| minta.framer.ai | Black page. One diagonal beam of orange-red light across a hand. Headline at the bottom centre, thin weight, two short lines, one small pill button. First screen holds about 25 words. Next: a row of four big numbers in mono type, one short label each. Then a small mono caps label, a heading, and glass cards that each show one piece of real interface with a two-line caption. | The skeleton: one image, one sentence, one button; then a number row; then cards that show the thing instead of describing it. |
| flip-assistant.vercel.app (Johnny's own, built from Minta) | Same skeleton with a gold beam and 3D coins. Headline is Inter, weight 400, about 42 px at this width. Number row: "0 gp / 5 min / 2021 / 1 key". Three cards under "One panel. Three jobs." Four numbered steps in a stepped row. A data board that looks like the real product. | Proof that the Minta skeleton survives a different subject. The number row maps straight onto our results. |
| linear.app | Background rgb(8, 9, 10). Headline Inter, weight 510, 56 px, letters pulled tight (about -1.2 px), left aligned. The hero picture is the real product screen, large. Lower sections: one big sentence in two tones (white, then grey), three cards with line drawings. Almost no colour. | "The product is the hero." Two-tone sentences instead of a heading plus a paragraph. |
| raycast.com | Background rgb(7, 8, 10). Centred headline, weight 600, 64 px, over one bold graphic (red diagonal bars of light). Next screen: the real product window, with a small row of icon tabs under it that switch the demo. One sentence of copy: a bold lead-in, the rest grey. | Icon tabs under one stage is the same job as our "Two ears / My ear" switch. |
| framer.com | Headline on the left, two buttons, then one large "stage" card playing a product demo. | One stage, not two panels side by side. |
| higgsfield.ai | On this date the home page is a Black Friday sale: neon lime on black, a countdown, a grid of video cards with the caption on the picture and small label chips. Busy today. | Only the durable parts: one accent colour, media cards with a caption on the image. |
| krea.ai | Looks like an app: a left rail of tools, large media cards. | Little for a story page. |
| ciechanow.ski/sound | Light page, one text column 704 px wide, body text 19 px. About 8,000 words with 48 interactive figures: each idea gets one small figure right under the paragraph that introduces it, often with one slider. No cards, no boxes, no legend. | The pattern for "How it works": one idea, one small thing to drag, then the next idea. |
| hrtf.tlifschitz.com (Spatial Audio Lab) | A full-screen 3D head. One line, "Put on your headphones", and one button. Nothing else until you start. | The headphone moment as its own screen. |
| switch-on-life.com/overview | One step per full screen, a dot pager with eight steps, one action per screen ("Meet Your Friends. Listen carefully!"). | A listening task split into steps, one thing to do at a time. |
| darkmodedesign.com (gallery, first page) | Entries seen: Vitrine, Lime Duo, Mathias Adolfsson, Rain, Ambre Fumé, LUSTRE, Stash, Ardor Retro Computer, Jana. Most show one huge word or one object and little else. | Confirms the common thread: one big thing per screen. |

Not opened this round (less relevant to a story page, or product dashboards): awwwards, land-book, framer.com/gallery,
mobbin, refero, tradingview, blitz.gg, flippingcopilot.

## The current page next to these (https://johnfok119.github.io/second-ear/)

- **First screen is full.** Title, tagline, three nav pills, a section label, a heading, a paragraph, two number
  boxes, a source line, the "Hear it with my ear" heading with its sentence, a play button and a switch, a status
  line, a headphones note in a box, then the next heading and paragraph. Minta's first screen is one picture, one
  sentence and one button.
- **One colour does every job.** Amber is the title, the labels, the numbers, the buttons, the borders and the dots.
  The references use near-black, white text and one accent that appears in a few places only.
- **Boxes inside boxes.** Almost every block has a border and a tinted fill.
- **The main exhibit is small and late.** The 3D café is two panels side by side in the middle of the page, each
  under half the width, in brown on brown.
- **A legend explains the picture.** Seven marks are explained in a list under the café, followed by two paragraphs
  of small print. The references label things where they are.
- **Heavy type.** Many bold runs inside body text; the references use lighter weights and far fewer words.

## Patterns that fit second-ear (options)

| Option | Borrowed from | What it would mean here |
|---|---|---|
| **A. One light idea** | Minta, Raycast, Flip Assistant | The top of the page is one image about direction: light (a sound) arriving at a head from one side. Two short headline lines. Then the number row: 3.7° and 22.4° (people, two ears and one), 0.479 and 0.207 (the model on real recordings, two ears and my ear). |
| **B. The café is the stage** | Linear, Raycast, Framer | The 3D café fills the width as one stage near the top, with a "Two ears / My ear" switch under it like Raycast's icon tabs, and marks labelled in place instead of a legend. |
| **C. Explainer column** | Ciechanowski | "How it works" becomes one column of short paragraphs, each with one small figure to drag. The existing drag-the-sound toy already fits. |
| **D. One step per screen** | Spatial Audio Lab, Switch on Life | The listening part gets its own screen: "Put on your headphones", then two ears, then my ear. |

These combine: A or B for the top, D for listening, C for how it works, and Minta-style cards for the results story
(the fine-tune before and after from Phase 8 is a natural pair of numbers on one card).

Earlier research on what to show, not how it looks, is in the day log `personal\private\projects\second-ear\logs\11-07-2026.md`
(hearing-loss simulators, explainers, direction visuals).

## Questions for Johnny

1. Top of the page: A (one light idea) or B (the café as the stage)?
2. Same family as the Flip Assistant site (black, one accent, Minta skeleton), or clearly different because both sit
   in the portfolio?
3. Still unanswered from 2026-11-05: what bothers you most about the current page (the look, the words, or that
   visitors do not get it)?
