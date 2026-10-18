# Validation snapshot

## Output-clock verification revision

A user reported that ticking only "change pitch" failed the job with
`Rendered waveform failed structural verification`, while ticking "change timing"
as well succeeded. The order-independence was the clue: the two cases differ only
in the timeline the renderer produces.

When the timing is not changed the output keeps the **student's** timeline, so a
correct render is exactly as long as the student recording. Verification, however,
measured the output against `teacher_analysis["sample_count"]` everywhere. On the
reported material the output was 329,466 samples -- exactly the student's length --
and was compared against the teacher's 322,877, so `sample_count_exact` was False
and the structural check failed a waveform that was correct.

The revision makes the clock explicit rather than assumed:

- `verify_output` takes the clock (default: what `target_pitch.npz` declares via
  `timing_changed`) and sizes the output against the teacher's or the student's
  analysis accordingly. The `student-analysis` path is now passed by `run_job`, and
  a student-clock verification without it raises instead of guessing.
- Teacher-timeline metrics are computed only on the teacher clock. On the student
  clock they are absent, with a limitation saying so, because the teacher's
  timestamps do not describe the same moments; the render-target comparison, which
  is clock-consistent, still runs.
- The YIN diagnostic drops its `vs_teacher` comparison on the student clock and
  reports `clock`; the planned-boundary seam diagnostic is skipped, since those
  boundaries are teacher times.
- `verify_timing` refuses the comparison on a student-clock output and records the
  reason, instead of subtracting teacher timestamps from unrelated output ones.
  `run_job` then reports the job as unresolved and surfaces the reason, rather than
  reading coverage that was never measured.
- The report and the window label the timeline: the pitch chart heading, the
  correspondence columns ("Teacher time" / "Student time"), and the timing row all
  now say which clock ran.

Verified end to end on the same analyses for all four option combinations, with the
timing both kept and changed:

| Timing | pitch_mode | map_teacher_pitch | Clock | Structural check |
| --- | --- | --- | --- | --- |
| kept | off | off | student | passed |
| kept | teacher | off | student | passed |
| kept | teacher | on | student | passed |
| kept | adaptive | - | student | passed |
| changed | off | off | teacher | passed |
| changed | teacher | on | teacher | passed |

The student-clock rows are the case that previously raised. All six now finish as
`completed_with_unresolved` for their own substantive reasons, with the timeline
recorded in `verification.json` (`provenance.output_clock`, `audio.output_clock`).

## Level measurement and the adaptive contour revision (current)

*This is the most recent revision; the sections below it are kept as history.*

A user reported that adaptive mode "measures badly and often goes out of tune,"
and proposed that the cause was the key mapping: "teacher in D major, student in C
major, so map the teacher's D-major pitch into the student's C major."

The key hypothesis was tested first and did not hold. Using Krumhansl-Kessler key
profiles on the frame pitch-class histograms, the teacher is **F major** and the
student is **C major** -- five semitones apart, which agrees exactly with the
measured register difference of -500 cents. The register relationship was already
right. Detecting key per 60-second section also gave no single key (C, D, C, D, G
minor across the piece), and the mean distance from each singer's words to their
best key was 24.8 and 26.6 cents, near a quarter tone, which is not a scale.

The real cause was how a word's level is measured. The within-word pitch span
(p5 to p95) has a median of **243 cents** on the teacher and **199 cents** on the
student: most "words" carry a glide, a vowel onset and a release, so a word is not
one pitch. Taking the median of the whole word therefore mixes the gesture into
the note, and how much depends on the word's shape and on where the alignment put
its boundaries. The same word measured two reasonable ways disagreed by more than
25 cents for 30% of the teacher's words.

`_sustained_cents` reads the note instead: the peak of the word's own pitch
histogram is the note the singer settled on, and the frames within 50 cents of that
peak are averaged back in. Validated against **GAME's independent note extraction**
on the same words, this is a real improvement rather than a preference:

| Measurement (473 words, long pair) | Whole-word median | Sustained note |
| --- | ---: | ---: |
| Median error against GAME's note | 12.4 cents | **5.7 cents** |
| Words within 25 cents of GAME's note | 66.4% | **85.2%** |
| Words within 50 cents | 81.0% | **90.7%** |
| Share of words where this estimate is closer | 36.8% | **63.2%** |

The included short pair shows the same ordering (18.9 -> 6.7 cents median error,
70.6% -> 82.4% within 25 cents).

### What did not improve, and is reported as such

The change makes the estimator more accurate; it does **not** make the applied
per-word movement smaller, and the difference is worth stating plainly:

| Per-word movement (473 words) | Whole-word median | Sustained note |
| --- | ---: | ---: |
| Register offset | -500 cents | -500 cents |
| Median absolute move | 16.9 cents | **9.9 cents** |
| p95 absolute move | 147.7 cents | 201.8 cents |
| Words moved more than 25 cents | 191 | **130** |
| Words moved more than 50 cents | 108 | 106 |

The median move and the count of large moves both fall, but the p95 rises. The
more accurate estimate is also a less smoothed one: a median compresses the
distribution toward the register, so some of what looked like agreement was the
estimator averaging differences away. The remaining spread is a real difference
between the two singers' intonation, not measurement noise. Reducing it further
would mean moving words the way a correction tool does, which is what the
`adaptive` setting already does and what `teacher` mode deliberately does not.

Because the per-word estimate now carries a confidence, each region records
`teacher_sustained_share` and `student_sustained_share`: the share of the word the
held note actually covers. A low share marks a word that was mostly gesture, so its
level is a weaker observation.

### The adaptive contour

The contour checkbox is now available in adaptive mode too, which the user asked
for. The teacher's contour already carries the teacher's note levels, so
transposing it by the register difference puts every note where per-word placement
was aiming -- and keeps the teacher's own slides and vibrato. That combination
needs no per-word measurement at all, so it reports `placement: constant`:

| Pitch mode | Contour | Policy | Placement |
| --- | --- | --- | --- |
| off | - | `none` | none |
| teacher | off | `student_contour_teacher_register` | constant |
| teacher | on | `teacher_contour_teacher_register` | constant |
| adaptive | off | `student_contour_student_register` | per word |
| adaptive | on | `teacher_contour_student_register` | constant |

All three settings were re-rendered end to end on the included short pair: each
finishes with the clock the timing option implies, the structural checks pass, and
the pitch policy and placement match the setting (verified: `off` -> `none`/`none`,
`adaptive` -> per word 8 of 17 words moved, `adaptive` + contour -> constant with 0
per-word moves).

A limitation of the estimator is recorded in the test suite rather than papered
over: when a glide takes longer than the note it lands on, the histogram peak sits
inside the glide, so the reading lands between the glide and the note. That shape
is not separable by this method, which is why the per-word results are reported
with their own measured error.

## Adaptive pitch mode revision (superseded)

A user asked for a third answer to the pitch question: "if adaptive mode is
selected, the pitch is arranged to look like the teacher's inside the student's
range". The two states the switch had -- leave it alone, or force it to the
teacher -- could not express that, because a student who sings the song a fifth
lower is not making a mistake, and dragging them up to the teacher's level changes
who they sound like.

The pitch control is now a three-way switch plus the unchanged contour checkbox:

| Pitch | Contour | Result |
| --- | --- | --- |
| off | - | The student's pitch, completely unchanged. |
| teacher | off | The student's own contour, moved as one constant into the teacher's register. |
| teacher | on | The teacher's contour at the teacher's absolute pitch. |
| adaptive | - | The student's own contour, with each word placed on the teacher's note level inside the student's own register. |

### What adaptive mode does

The register offset is measured exactly as before: the median level difference over
matched regions voiced on both sides, quantized to whole semitones. It anchors the
teacher's note levels inside the student's register. Per region, with
`student_level` and `teacher_level` the measured medians and `offset` the measured
register difference, the shift is

    shift = (teacher_level + offset) - student_level
          = offset - (student_level - teacher_level)

so the word's median lands exactly on the teacher's note transposed into the
student's own register. Two consequences follow directly and are the reason the
setting behaves the way it does:

- A student who already sings the right melody uniformly below the teacher needs
  **no correction at all**: every shift is zero. Singing in your own range is not
  an error, and adaptive mode is the setting that knows it.
- A word that is genuinely wrong is moved by exactly its own error, and by nothing
  else.

The per-word path is the one that was removed in the previous revision because it
ran unconditionally. It is back, gated to this policy alone, with the guards the
earlier incidents argued for:

- A shift beyond `_MAXIMUM_NOTE_SHIFT_CENTS` (600 cents, a tritone) is refused and
  reported: a difference that large is more likely an octave error in pitch
  tracking than the interval sung.
- A residual below `_NEGLIGIBLE_NOTE_SHIFT_CENTS` (10 cents) is left alone, so a
  word that is already on its note is not detuned.
- A region voiced on only one side has no note to reach and is left alone.
- Without a measured register there is no anchor, so every word is left alone and
  the render says so rather than guessing.
- Each word's shift is held constant across the word, so the student's own slides
  and vibrato keep their shape; the change between two words is ramped over 30 ms
  instead of stepping, and only there.

`off` maps to the `none` policy, which is now the **only** policy that does not
change the pitch. A timing-only job therefore still cannot move a single frame, no
matter what the other options say.

### Measured on the included long pair

The student sings 496.2 cents (about five semitones) below the teacher, corroborated
by 473 comparable regions, so this is the material where the difference between the
three settings actually shows. Same analyses and `alignment.json` for every column;
only the pitch setting changed.

| Measurement | `off` | `teacher` + contour | `adaptive` |
| --- | ---: | ---: | ---: |
| Policy | `none` | `teacher_contour_teacher_register` | `student_contour_student_register` |
| Placement | none | one global constant | per word |
| Register offset | not measured | −500.0 cents (raw −496.2) | −500.0 cents (raw −496.2) |
| Words moved | 0 / 473 | 0 (contour replaced instead) | **305 / 473** |
| Target median F0 | 191.34 Hz | 247.31 Hz | **191.30 Hz** |
| Median \|error\| vs the teacher's note levels in the student's register | 496.09 cents | 500.21 cents | **3.13 cents** |
| Words within 50 cents of their target note | 0.0% | 0.0% | **98.3%** |
| Structural checks | passed | passed | passed |

Adaptive placement puts the same material 305 words closer to the teacher's notes
(496.09 → 3.13 cents median, 98.3% within 50 cents) while the median F0 stays at
191.30 Hz against the untouched render's 191.34 Hz -- musically the same register,
so the correction does not change who the student sounds like. The `teacher`+contour
row is correct rather than wrong: it copies the teacher's absolute curve by
definition, so against a register-adjusted reference it measures the intended
five-semitone transposition, not error. Row by row, adaptive improved 472 of the 473
words against that reference, left none unchanged, and worsened one.

On the included short pair the two singers are already 4.1 cents apart, which
quantizes to zero, so no register move is required at all; adaptive still placed 14
of the 17 words onto their target note inside the student's own register, and `off`
left the median exactly where it was sung (309.09 Hz in both cases).

| Measurement | `teacher` (contour off) | `adaptive` |
| --- | ---: | ---: |
| Policy | `student_contour_teacher_register` | `student_contour_student_register` |
| Placement | one global constant | per word |
| Words measured / moved, in-tune student | 3 / 0 | 3 / **0** |
| Words measured / moved, one word 300 cents sharp | 3 / 0 | 3 / **1** |
| Resulting level of the sharp word | unchanged | its own in-tune level |

The in-tune row is the whole point: the in-tune student is not moved, while the
`teacher` mode still moves the melody into the teacher's register by definition.

### Consequences for verification and the report

- `target_pitch.npz` carries `note_shift_applied_count`, so a verification report
  knows whether the words were placed individually.
- When they were, the register-shifted teacher curve only *approximates* the aim, so
  the report adds a limitation saying so and points at `actual_vs_render_target`,
  which is the authoritative per-frame comparison.
- The report's register section states how many words were moved, and
  `render.json` records the placement kind (`none`, `constant` or `per_word`), the
  per-region shift list, and the measured/moved counts.
- The window shows the moved-word count next to the register offset and disables the
  contour checkbox in `off` and `adaptive`, where it has no effect, instead of
  offering a choice that does nothing.

All six settings (three modes plus the contour checkbox where it applies) were
re-rendered on the included pairs: each finishes with the clock the timing option
implies, the structural checks pass, and the pitch policy matches the switch. The
299-test suite covers the placement arithmetic, its guards, the ramp, the clock
selection of the shift track, and the end-to-end renderer behavior in both the
in-tune and the wrong-note cases.

## Explicit-pitch-options revision (superseded by adaptive mode)

A user reported that a job with only "change timing" ticked had also changed the
pitch. It had: with both pitch options off the renderer still carried a per-word
`shift_cents` array, so a timing-only job moved 185 of 491 output frames, up to
**499.62 cents** (a tritone) on the word `我`.

The cause was a semantic mistake in the option mapping, not a numerical one. The
two pitch options answer "whose register" and "whose contour"; neither authorizes
a per-word pitch edit. The correction of individual wrong notes was being applied
unconditionally, as if it were part of the timing correction, and it also
contradicted the option's own promise: a student who simply sings a phrase
differently had those notes rewritten without being asked for.

The revision removes the per-word path entirely:

- `change_pitch` off + `map_teacher_pitch` off leaves the pitch exactly as sung.
  The policy reports `applied: false` and a reason, and the renderer copies the
  student's sampled F0 through untouched.
- The only pitch move that exists is one global constant,
  `contour_shift_cents = target register level - chosen contour's own level`. It is
  zero in two of the four combinations, which is what makes those two either
  pitch-preserving or absolute.
- Register measurement is unchanged and still required before a move is applied:
  the median difference over matched regions voiced on both sides, quantized to
  whole semitones, needing at least two comparable regions, refusing a
  measurement beyond two octaves.
- Per-region levels are still reported, now as evidence for the estimate rather
  than as a target for per-word edits.

Measured on the reported material (same analyses and `alignment.json` for both
columns; only the renderer changed):

| Measurement | Before the fix | After the fix |
| --- | ---: | ---: |
| Output frames compared against the student's own pitch | 491 | 491 |
| Maximum deviation from the student's own pitch | 499.62 cents | **0.00 cents** |
| Frames deviating more than 30 cents | 185 | **0** |

The four combinations were then re-measured on the long pair, where the student
sings 496.2 cents below the teacher, so a register move is actually required:

| change_pitch | map_teacher_pitch | policy | Applied move | Target median F0 |
| --- | --- | --- | ---: | ---: |
| off | off | `student_contour_student_register` | 0.0 cents | 191.3 Hz (student's own) |
| off | on | `teacher_contour_student_register` | −500.0 cents | 185.3 Hz |
| on | off | `student_contour_teacher_register` | +500.0 cents | 255.4 Hz |
| on | on | `teacher_contour_teacher_register` | 0.0 cents | 247.3 Hz |

The two zero-move rows are correct rather than missing data: the first keeps the
student's own pitch, and the last uses the teacher's contour at the teacher's own
level, which is the historical absolute behavior.

## Register and contour options (superseded by adaptive mode)

Earlier revisions had one pitch behavior: the teacher's **absolute** F0 was
copied onto the student. On the difficult pair the student sings a median 445.9
cents (about 4.5 semitones) below the teacher, so that forced the whole corrected
vocal up to the teacher's register. Nothing in the report said so, and the
teacher-referenced pitch error looked excellent (6.82 cents median) precisely
because the output *was* the teacher's pitch.

Two independent options now select the pitch target, and `change_pitch` is what
decides whose register the result sits in:

| `change_pitch` | `map_teacher_pitch` | policy | Result inside matched regions |
| --- | --- | --- | --- |
| off (default) | off (default) | `student_contour_student_register` | Student's own contour and register; each word moved onto the teacher's melody level. |
| off | on | `teacher_contour_student_register` | Teacher's continuous contour, moved into the student's register. |
| on | off | `student_contour_teacher_register` | Student's own contour, moved into the teacher's register. |
| on | on | `teacher_contour_teacher_register` | Teacher's contour and register as analysed (the earlier behavior). |

`correction/register.py` supplies the mapping:

- The **register offset** is the median level difference over matched regions
  voiced on both sides, quantized to whole semitones so the rendered melody stays
  in a well-tempered key instead of being detuned by a fraction of a semitone. It
  is only used by the two student-register policies; a teacher-register render
  targets the teacher's absolute levels directly.
- Each matched region then carries the difference between its target level and
  the student's own level there. Levels are robust per-region medians, so one
  wrong note cannot dominate a region.
- The offset needs at least two comparable regions. With one word there is no way
  to separate "this singer is a fifth below throughout" from "this one word was
  sung low", and treating the difference as a register would absorb it into the
  offset and cancel the very correction that word needs.
- Per-region shifts are clamped to one octave, a measured offset beyond two
  octaves is refused as a likely octave-tracking error, and residuals below 10
  cents are treated as already correct rather than detuning every word.
- For the teacher-contour policies the per-word shift is not applied at all, and
  each region records `teacher_contour_used` so the number cannot be misread.

### The difficult long pair

The student sings 445.9 cents below the teacher; the exported register offset is
**-500.0 cents** (raw -496.2), corroborated by 473 comparable regions, with 306 of
473 words moved onto a different note and none clamped. The same `alignment.json`
and analyses were used for every column.

| Measurement | Archived absolute render | New default (`student_contour_student_register`) | `change_pitch` on (`student_contour_teacher_register`) |
| --- | ---: | ---: | ---: |
| Exported target median F0 | 247.3 Hz | 191.2 Hz | 255.4 Hz |
| Pitch error vs the **aimed** target, median | 6.79 cents | 4.65 cents | — |
| Pitch error vs the register-matched teacher, median | not reported | 29.25 cents | — |
| Pitch error vs the teacher's **absolute** pitch, median | 6.82 cents | 497.21 cents | — |
| Teacher-voiced recall | 0.9279 | 0.9338 | — |
| Structural checks / exact sample count | passed | passed | passed |

The new default places the corrected voice at 191 Hz, in the student's own
register, and the 497-cent "error" against the teacher's absolute pitch is the
intended five-semitone placement rather than a defect. Against the pitch the
renderer actually aimed at it is *more* accurate than the archived absolute
render (4.65 versus 6.79 cents median), so keeping the student in their own
register costs no accuracy. Turning `change_pitch` on reproduces the older
behavior of moving the student up into the teacher's register, at 255 Hz.

Because agreement with the teacher's absolute pitch no longer measures accuracy
under the student-register policies, `verification.json` adds a register-matched
comparison whenever the offset is not negligible, plus a limitation and a warning
saying which reference each number used.

### The included short pair

Both singers are already in the same register there, so the offset is small and
the register decision is nearly a no-op: raw -4.1 cents, quantized to 0.0,
corroborated by 17 comparable regions, with 14 of 17 words carrying a median
34.5-cent shift to correct individual wrong notes. Pitch error against the aimed
target is 5.64 cents median. A same-register pair is therefore not measurably
disturbed, which is the evidence that the offset estimate does not invent a
transposition where none exists.

### Independent review

No independent subagent review of this revision was completed: five spawn
attempts across three agent types returned a provider error before doing any work.
The numbers above come from the pipeline's own verification pass and from direct
comparison of the exported targets, not from a second implementation. Listening
assessment remains outstanding, and it is the only way to judge whether keeping
the student in their own register sounds better than moving them to the teacher's.

## Consonant-preservation revision

The previous continuous-mapping revision removed the artificial silence but
stretched consonants to cover teacher inter-word rests. Near 47.4 s the student
`z` fricative (47.494989-47.604989, 110 ms) was mapped across roughly 280 ms with
a local rate of 3.19x, and its onset was pulled about 200 ms early. A stretched
fricative is a static hiss; listeners hear it as a stuck machine. The current
revision bounds that failure mode directly.

`inference/correction/durations.py` partitions each transition window into
phones and the rests between them, then budgets target time per segment:

- A consonant may be compressed but **never** exceed the length the student
  actually sang. Its upper bound is its own sung duration in every tier.
- The searched warp is not trusted for a consonant. Its cost is dominated by
  spectral continuity, under which stretching a stationary fricative looks
  almost free, so the planner's own vowel/consonant budget supplies the target.
- Vowels absorb remaining time; genuinely quiet rests flex first. An unfittable
  remainder is handed out in the order quiet rest, span, vowel, consonant and is
  reported through `residual_unresolved` rather than hidden inside a consonant.
- Segment boundaries are pinned on both timelines and joined by a
  shape-preserving monotone cubic, so the rate is blended across joints. That
  rate blending is the warp-domain crossfade; a tempo step at a phone or word
  joint would be audible even when source position is continuous.
- `renderer._crossfade_splices` additionally equal-power blends mel across a
  window edge where the sliced map left a genuine source discontinuity.

Measured on the difficult pair, using the exported `target_pitch.npz` mapping
and the student phoneme timeline:

| Consonant duration check | Previous revision | Current |
| --- | ---: | ---: |
| Consonants measured (within one region) | 466 | 466 |
| Maximum output / sung duration | 3.19x (at the 47.4 s `z`) | 0.995x |
| Consonants above 1.02x | many | 0 |
| Maximum local stretch anywhere in a refined window | 5.41x | 2.43x |
| Refined windows above 2x local stretch | 50 of 78 | 6 of 78 |
| Refined windows above 3x local stretch | 9 of 78 | 0 of 78 |

The 47.4 s window now plans `ir` 0.240 s -> 0.306 s and `z` 0.110 s -> 0.110 s:
the 220 ms rest is carried by the vowel at 1.27x while the fricative is
unchanged. All 73 constrained windows satisfied their budget at the preferred
tier with zero residual. Nine windows carry no phoneme labels and therefore
leave the searched warp untouched, which is reported as
`duration_constraint.applied = false` rather than guessed at.

| Current metric | Short pair | Difficult long pair |
| --- | ---: | ---: |
| Samples at 44.1 kHz | 322,877 | 12,806,588 |
| Jointly voiced RMVPE median / p95 error | 5.97 / 69.52 cents | 6.82 / 66.55 cents |
| Measured teacher-voiced recall | 99.10% | 92.79% |
| Reliable exact words / teacher units | 16 / 17 | 431 / 477 |
| Reliable words with both bounds <=50 ms | 16 / 16 | 405 / 431 |
| Reliable onset / offset MAE | 4.38 / 3.13 ms | 5.75 / 14.55 ms |
| Constrained transition windows | 2 | 73 of 82 |
| Splice crossfades applied | 0 | 3 |
| Finite audio, exact teacher count, no clipping | Pass | Pass |

Matched comparison on the 426 words reliable in both the previous and current
long runs: onset MAE improved from 10.89 to 5.72 ms and onset agreement within
50 ms from 411/426 to 423/426. Offset MAE moved from 12.37 to 14.66 ms and
within-50 ms agreement from 407/426 to 402/426; 143 words got better and 144
worse, so the offset change is a wash rather than a uniform regression. The
largest offset increases (69.15 s `见`, 183.375 s `在`, 47.080 s `失`) are words
whose student vowel is held across a teacher breath that has no matching
aperiodic source material; that residue is reported, not solved.

Both runs keep status `completed_with_unresolved`. Current WAV SHA256:

- Short: `8c3402d5b3fe85c2...`
- Long: `145995bf95910c3c...`

The criticized revision is preserved for comparison in
`outputs/long_hard_continuous/revision_stretched_consonant` (WAV SHA256
`580cfcc7e4a9f725833137172188b85dc8e67245a781ddf32fb74ee23d65af23`). Listening
excerpts for 46.8-48.2 s are regenerated under
`outputs/long_hard_continuous/transition_excerpts`.

CPU tests: 202 passing, compilation passed. This revision has **no independent
subagent review**: four spawn attempts across three agent types returned
provider errors with no output, so the verification above is the main agent's
own measurement. That is weaker evidence than the earlier revisions received
and should be treated as such.

The remaining known limitations from the continuous revision still apply:
teacher aperiodic material near 138, 190, and 227 s has insufficient sustained
student noise, so periodic vowel texture is still mapped there. Those windows
are recorded by `source_acoustic_mismatch`. Consonant length is now enforced,
but that does not create missing source material or establish naturalness.

## Continuous acoustic mapping revision

Japanese integration was added after the continuous-mapping audio runs below.
Those WAVs and their provenance remain records of their saved code revision;
they have not been regenerated with the Japanese integration. Japanese resources,
entry points, and mora correspondence have implementation/unit checks, but no
Japanese audio end-to-end or singing-quality validation has been performed.

The current implementation uses separate `outputs/short_continuous` and
`outputs/long_hard_continuous` directories. The original baseline and earlier
tail-only output directories remain preserved. This saved revision had 134 passing
CPU behavioral, contract, and regression tests. Compilation passed for
`application`, `scripts`, and `inference`. Both complete CLI runs finished with
waveform pitch and output-only ASR/TiFA timing verification; both retain the
status `completed_with_unresolved`. GPU stages ran sequentially.

| Current continuous-mapping metric | Short pair | Difficult long pair |
| --- | ---: | ---: |
| Samples at 44.1 kHz | 322,877 | 12,806,588 |
| Jointly voiced RMVPE median / p95 error | 5.76 / 69.15 cents | 6.89 / 68.93 cents |
| Measured teacher-voiced recall | 99.10% | 92.80% |
| Reliable exact words / teacher units | 16 / 17 | 433 / 477 |
| Reliable words with both bounds <=50 ms | 16 / 16 | 399 / 433 |
| Reliable onset / offset MAE | 6.25 / 3.13 ms | 10.79 / 12.61 ms |
| Continuous acoustic transition windows | 2 | 82 |
| Reference-relative energy diagnostics | 2 | 80 |
| Frames with acoustic energy attenuation | 0 | 2,309 |
| Finite audio, exact teacher count, no clipping | Pass | Pass |

Independent CPU inspection checked the input, runtime-code, resource-manifest,
alignment, exported WAV, target NPZ, and verification provenance. The saved
timing comparisons were also recomputed from the stored analyses and matched
the reports. All 84 refined transition windows are finite, strictly increasing,
and bounded by their fixed outer anchors. Four explicit missing/rejected lyric
barriers in the long output remain exactly zero. The short output has no such
explicit lyric barrier: its four unresolved voiced-gap plan labels do not mean
that the final waveform is zero, because eligible acoustic context covers them.
The machine-readable checks and residual limitations are recorded in
`outputs/long_hard_continuous/final_artifact_audit.json`. Its historical-path
notes identify archived reports whose original absolute paths now point to the
current output; use each archived WAV and its recorded hash for comparisons.

Current corrected WAV SHA256:

- Short: `8043092f66b8256a4b141466cab9c055c1fe876450178f4094064540f5fc598f`.
- Long: `580cfcc7e4a9f725833137172188b85dc8e67245a781ddf32fb74ee23d65af23`.

The preceding tail-only approach retained short releases but still discarded
most interword audio. In the difficult recording, the target interval
47.36–47.58 s between `失` and `在` had no dedicated source gap: their source
word estimates touch at 47.494989 s. A 40 ms held tail left 47.40–47.58 s
exactly zero. RMVPE labeled most of this teacher interval unvoiced, while the
waveform contained strong aperiodic audio:

| Historical tail-only audit interval | RMS | Exact-zero samples |
| --- | ---: | ---: |
| Teacher 47.40–47.58 s | -28.19 dBFS | 0% |
| Corresponding student 47.495–47.605 s | -30.04 dBFS | 0% |
| Tail-only output 47.40–47.58 s | Zero | 100% |

The source interval overlaps the following `z` consonant according to TiFA.
Its phonetic label and edges are model estimates, but the nonzero waveform is
direct evidence that pitch-unvoiced does not imply acoustic silence. A 10 ms
whole-track audit found 15.13 s of exact-zero nonlexical output inside the
lyric extent while teacher RMS exceeded -45 dBFS. This threshold identifies
audible-energy candidates, not confirmed words or a perceptual quality score.

The current design retains a continuous source trajectory through eligible
consecutive correspondences, using fixed outer phone anchors and local
acoustic timing. Sustained broadband-noise onset landmarks can align aperiodic
transitions; each side uses constrained DTW or continuous linear interpolation.
Long real source gaps use a bounded linear map. This can redistribute the
neighboring phone durations without promoting acoustic coverage to semantic
success. Missing/rejected lyrics and intervening unmatched source words remain
barriers. Original lyric annotations and semantic masks remain fixed, while
actual `source_times` and sampled source voicing can change inside refined
windows. See `architecture.md` for limits and diagnostic fields.

In the reported 47.373–47.608 s interval, the historical tail-only output had
about 180.4 ms of exact-zero samples. The current output has none, with RMS
0.02890 versus teacher RMS 0.0404. The 46.8–48.2 s teacher, baseline, and current
listening excerpts are saved under
`outputs/long_hard_continuous/transition_excerpts`. Nonzero samples demonstrate
that the previous hard mute was removed; they do not prove naturalness.

The full-track 10 ms grid audit in `transition_comparison.json` uses teacher
20 ms RMS >=0.003 to identify audible nonlexical-gap candidates. Of 2,292 such
frames, 1,714 were all-zero in the tail-only output and 92 are all-zero now
(74.78% -> 4.01%). The low-energy fraction changed from 90.79% to 43.24%; this
uses an absolute RMS floor of 0.0003 or 10% of teacher RMS. These are waveform
coverage diagnostics, not semantic coverage or listening scores. The short
pair's remaining 19 audible all-zero grid frames are in its untranscribed
opening; continuous interword mapping does not supply missing lyric identity.

Waveform periodicity landmarks now map transitions near 29, 109, 183, and 249 s
to real student aperiodic material. Reference-relative waveform RMS ceilings
also reduce inappropriate sustained sound during genuine teacher quiet gaps.
For example, the inspected windows near 42.51, 133.79, and 209.46 s changed from
RMS 0.04046 / 0.11412 / 0.07155 to 0.00380 / 0.02519 / 0.01956. The centers of
the latter two gaps are quieter than their shoulders; the entire gap is not
claimed to match teacher energy. Gain is smoothed, never amplifies, and is
applied once after vocoder synthesis.

Timing remains a regression relative to the tail-only revision: its long
measurement was 424/430 reliable words within 50 ms, versus 399/433 now. On the
420 words reliable in both runs, the count changed from 414 to 389. ASR/TiFA
share models with input analysis, so these are estimated boundaries rather
than manual ground truth, but the regression must not be hidden by improved
acoustic continuity. Near 138, 190, and 227 s, mapped audio still includes
periodic vowel texture where teacher is aperiodic. The latter two source
intervals contain insufficient sustained noise; a brief burst is not stretched
into a long consonant. These residuals are recorded by
`source_acoustic_mismatch`, which cannot establish a lyric match or silently
change semantic support. The current revision does not establish naturalness
throughout the full recording.

Run the current implementation in its separate output directories:

```powershell
.\run.bat --output .\outputs\short_continuous --verify-timing
.\run.bat --teacher .\teacher_long_hard.wav --student .\student_long_hard.wav --output .\outputs\long_hard_continuous --verify-timing
```

When a directory already contains input analyses with matching file hashes,
add `--skip-analysis` to reuse them. Use a normal run after changing analysis
settings or algorithms. A current-code rerun does not reproduce a historical
renderer revision merely by selecting its old directory name.

## Historical tail-only boundary-release revision

The abrupt-cut baseline is preserved in `outputs/short` and
`outputs/long_hard`. The following tail-only revision and its verification
remain in `outputs/short_natural` and `outputs/long_hard_natural`. Every score,
hash, and unchanged-source-map claim in this section applies to that historical
revision only.

The original renderer floored mel features immediately at lyric-region ends
and applied a 5 ms fade entirely inside those regions. This removed natural
release material. The revision separates semantic anchors from bounded acoustic
context: attack <=20 ms, real source release <=60 ms, held terminal texture
<=40 ms. Held texture also follows teacher-relative release energy; context
feature extraction cannot read another source word through its STFT window.
Three actual source jumps in the long sample receive local mel blending.
Explicit missing/rejected lyric spans remain exactly zero. Unsupported gap
labels and planned semantic coverage remain unchanged.

An intermediate 80 ms held-release experiment caused loud sustained tails and
measurable word-offset regressions. It was rejected in favor of the limits
above; selected artifacts remain under
`outputs/long_hard_natural/revision_held80` for diagnosis. They are not the
accepted tail-only revision. Shrinking held releases and adding reference energy
reduced the five inspected overextended tails by approximately 16–24 dB over
their first 40 ms. These comparisons are against that intermediate experiment,
not against the original zero-tail baseline.

Tail-only checks: 103 CPU tests passed, compilation passed, and both full CLI runs
completed with waveform pitch and output-only ASR/TiFA timing verification.
Independent CPU audit confirmed that revision's code/audio/analysis provenance, exact
output counts, unchanged semantic support/source maps/F0 inside support,
context limits, source-word protection, and exact rejected-word mutes.

| Historical tail-only metric | Short pair | Difficult long pair |
| --- | ---: | ---: |
| Samples at 44.1 kHz | 322,877 | 12,806,588 |
| Jointly voiced RMVPE median / p95 error | 5.53 / 69.17 cents | 6.82 / 68.46 cents |
| Measured teacher-voiced recall | 99.10% | 92.58% |
| Reliable exact words / teacher units | 16 / 17 | 430 / 477 |
| Reliable words with both bounds <=50 ms | 16 / 16 | 424 / 430 |
| Reliable onset / offset MAE | 3.75 / 3.13 ms | 5.65 / 7.47 ms |
| Acoustic contexts | 6 | 138 |
| Supported-interior RMS change from baseline | +0.0032 dB | +0.0026 dB |
| Finite audio, exact count, no clipping, exact rejected-word mute | Pass | Pass |

The three short phrase endings now contain decaying source release rather than
immediate zeros. Over all long support-island endings, aggregate RMS is .01905
in the first 20 ms and .00352 in the following 20–60 ms, versus zero in the
baseline. The script bounds these windows before the next supported word or
lexical barrier. Adjacent-sample differences at semantic edges can increase
because audio is retained there; this is not itself a click detector.

Timing has not improved universally: baseline long timing was 427/432 reliable
words within 50 ms with 5.59/6.68 ms onset/offset MAE. On the common 427-word
subset, final onset/offset MAE changed from 5.57/6.65 to 5.62/7.49 ms. The final
pass has six outliers, including five baseline outliers and one new TiFA onset
that absorbs about 60 ms of exact silence into the next word. This diagnostic
does not move the measured boundary or exclude it from the scores.

Exact 50 ms errors now use a 1 ns numerical comparison margin to avoid floating
point subtraction changing an inclusive tolerance. This is a scoring bug fix,
not an acoustic improvement. ASR/TiFA still share models with input analysis;
neither these metrics nor the boundary measurements certify perceived
naturalness. Conservative held-release limits can shorten a genuine long
vowel tail when no safe source continuation exists. Listen to the revised
files and compare with the preserved baseline.

Historical tail-only WAV SHA256:

- Short: `dca7aac1b0281565e1c7bceefa9299ecf590ae9d69e40e7bd80372299c2903ff`.
- Long: `a38334dca2a1a2795dbf516280bddc091191a3de5fef8dee2e4ed4b89a0aae9c`.

The retained `boundary_comparison.json` files describe this revision against
the baseline. Current-code reruns should use the separate directories above
to preserve these comparisons.

## Original baseline

Validated on 2026-10-15 with the project-local Python runtime, DirectML RMVPE,
and CUDA PC-NSF-HiFiGAN on an NVIDIA GeForce RTX 4070 Laptop GPU.

## Automated checks

- 310 CPU behavioral, application integration, feature-contract, and regression
  tests passed: `python/python.exe -X utf8 -m pytest tests -q`, which also runs
  the GUI tests headless.
- The pitch level estimator is checked against the shapes it must read (a held
  note, a small deviation, a glide into the note, an even vibrato, too few frames,
  and the glide-dominant shape it cannot separate). The GAME comparison above is a
  measurement taken for this revision, not an automated test.
- Each pitch setting is checked end to end through the renderer, including the
  newly available adaptive-plus-contour combination, and the option mapping is
  checked for every switch and checkbox state.
- Compilation passed: `python/python.exe -X utf8 -m compileall -q application scripts inference`.
- Final short and long CLI reruns completed, including exported-waveform pitch
  verification and output-only ASR/TiFA timing verification.
- Output audio hashes agree between waveform and timing verification. Job,
  verification, and alignment metrics agree with their corresponding artifacts.
  Recorded job and timing code fingerprints matched that baseline revision's runtime code.
- HTML content checks passed for both reports: all local audio/download links
  exist; both teacher and measured output pitch curves are present; final
  status, timing diagnostics, and model-estimate limitations are included.
- Browser screenshot inspection was unavailable because the browser tool
  blocks local `file://` navigation. No visual-layout pass is claimed.

The complete model and runtime migration audit is described in `resources.md`.
It verified physical copies, no symbolic/hard links or reparse points, matching
resource hashes, and project-local Python imports.

## Example results

| Metric | Short pair | Difficult long pair |
| --- | ---: | ---: |
| Output samples at 44.1 kHz | 322,877 | 12,806,588 |
| Teacher/output duration | 7.3215 s | 290.3988 s |
| Jointly voiced RMVPE median pitch error | 5.83 cents | 6.76 cents |
| Jointly voiced RMVPE pitch error p95 | 69.61 cents | 69.04 cents |
| Measured teacher-voiced recall | 98.92% | 92.10% |
| Planned semantic teacher-voiced coverage | 97.85% | 93.53% |
| Reliable exact words / all teacher units | 16 / 17 | 432 / 477 |
| Reliable words with both boundaries within 50 ms | 16 / 16 | 427 / 432 |
| Reliable-subset onset MAE | 4.38 ms | 5.59 ms |
| Reliable-subset offset MAE | 5.00 ms | 6.68 ms |
| Unsupported regions after voiced-run refinement | 4 | 141 |
| Finite PCM24 output, exact teacher count, no full-scale clipping | Pass | Pass |

All 16 reliably measured short-pair words have both boundaries within 20 ms.
The refined long plan contains 137 untranscribed voiced runs totaling 11.215 s
after clipping to lyric boundaries. The remaining unsupported regions are
lyric correspondence/quality problems. Splitting gaps changed report
granularity; all supported regions, local warp anchors, mel source times,
support masks, and exported target F0 were unchanged.

Both examples finish as `completed_with_unresolved`. Input lyric differences,
untranscribed activity, and missing or uncertain source material require
review. The long timing pass also has five boundary outliers above 50 ms and
45 teacher units outside the reliable exact-match subset.

## Interpretation limits

Pitch error covers jointly voiced frames; recall keeps missing teacher voicing
in its denominator. Planned semantic coverage is derived from the alignment
plan and is distinct from measured output voicing. RMVPE is used on both
teacher and output; the independent YIN diagnostic covers only selected stable
voiced windows.

The timing pass transcribes the exported audio without supplying teacher
lyrics or boundaries, then compares the resulting estimates with the teacher.
It shares ASR/TiFA models with input analysis and is not manually labeled
ground truth. Recognition differences can reflect model errors or rendered
lyric problems. Boundary uncertainty annotations do not remove observations or
improve the reported timing statistics.

No subjective listening pass, intelligibility certification, or objective
timbre-preservation score is claimed. Compare identity reconstruction with the
original student before attributing artifacts to the correction stage.
