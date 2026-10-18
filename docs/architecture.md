# CopyVocal architecture

## Objective and operating boundary

The teacher supplies the target clock and pitch trajectory. The student supplies the vocal spectral content. The implementation seeks to correct timing and melody while preserving the student's voice, and exposes uncertainty whenever source material or correspondence is insufficient. It does not synthesize new lyrics, clone the teacher's voice, or claim that a matching total duration proves word-level accuracy.

Inference runs from physical copies of models, a portable Python runtime, native backends, and FFmpeg under the project root. Original Vocal2Midi and vocoder source directories are provenance records rather than runtime paths. Operating-system drivers remain external. Existing analysis files contain absolute source paths. After relocation, `--skip-analysis` checks the explicitly supplied current inputs against the stored hashes and rebinds matching source paths, retaining the original paths for provenance. A normal analysis run is needed when input content or analysis algorithms/settings change.

The CLI reads `settings/default.json` before parsing stage options. `--config` selects another JSON file, and explicitly supplied flags override its defaults. Analysis chunk/context durations, minimum correspondence quality, maximum duration stretch (`maximum_stretch`), backend selection, and vocoder log floor are recorded in each job configuration. The bundled feature contract rejects a sample rate other than 44100 Hz.

## Process and memory boundaries

`application/copyvocal.py` orchestrates separate analysis, synthesis, and verification worker processes. Exiting a worker releases the associated DirectML/CUDA contexts before the next backend starts. This avoids retaining Qwen3-ASR, alignment, and synthesis resources concurrently on a constrained GPU.

`--verify-timing` adds another worker after waveform verification to recognize and force-align the generated audio. It is optional because it runs the lyric models again. No successful process exit is treated as a guarantee of complete correction: `completed_with_unresolved` covers unsupported/fallback regions, ambiguity, substitutions, local stretch extremes, or unmet optional timing criteria. A plan with no permitted synthesis material fails before rendering.

Within analysis, work is grouped by model stage across the two recordings. Qwen3-ASR performs recognition, TiFA aligns the resulting text, RMVPE estimates continuous F0 with a separate voiced mask, and GAME optionally extracts notes. `--no-game` changes available analysis diagnostics, not the F0 used for synthesis.

Audio is safely downmixed and resampled to 44.1 kHz. If channel averaging substantially cancels the vocal, the strongest channel is selected and reported. Invalid numerical input is rejected. The slicer chooses quiet cut points where available, otherwise bounded hard cuts, and includes 0.75 seconds of context by default. Chunk cores own disjoint original samples; midpoint ownership and deduplication reconcile context predictions. The default 18-second maximum includes context. Teacher and student chunks are analyzed independently; equal chunk numbers never imply lyric correspondence.

The vocoder extracts mel features on CPU in bounded STFT blocks. Synthesis uses context, overlapping chunks, crossfades, and coherent source phase. Full-duration feature/output arrays remain in host memory, so arbitrarily long recordings are not guaranteed to fit despite bounded GPU work.

## Analysis contract

Each `*.analysis.json` records:

```python
{
    "source": "absolute input path",
    "sample_rate": 44100,
    "sample_count": 322877,
    "duration": 7.3214739229,
    "words": [
        {
            "text": "lyric unit",
            "start": 0.52,
            "end": 0.70,
            "quality": 1.0,
            "phonemes": [{"text": "phone", "start": 0.52, "end": 0.56}],
            "issues": []
        }
    ],
    "pitch": {"times": [], "f0_hz": [], "voiced": []},
    "notes": [],
    "chunks": [],
    "completed_stages": ["lyrics", "pitch", "notes"],
    "warnings": [],
    "input_sha256": "...",
    "provenance": {}
}
```

All word/phone timestamps are absolute seconds on that recording's resampled clock. RMVPE timestamps preserve its own explicit time step; consumers must not assume that an RMVPE frame and a mel frame share the same center. Unvoiced frames have F0 zero and an independent `voiced` mask. This is a periodicity estimate: fricatives, stop releases, breaths, room sound, and some weak singing can remain audible without a detected F0. Zero F0 does not require zero waveform amplitude. The planner never uses F0 values as word-identity evidence.

TiFA is called with `preserve_gaps=True` and `raw_spans=True`. Degenerate or overlapping token predictions are reported instead of manufacturing tiny valid durations or filling every gap. The analysis quality score checks boundary structure, energy, phoneme availability, and warnings; it is not a calibrated ASR posterior. Recognition disagreement is therefore a suspected lyric mismatch or ASR error, not a conclusion about what the singer actually said.

Cache identity includes input content, models, relevant code, and configuration. Completed intermediate ASR/alignment/note chunks are written atomically and reused. Pitch uses an independently fingerprinted cache so a lyric-stage change can retain unchanged pitch analysis. `--skip-analysis` skips this work and loads output-directory analysis after checking input hashes; it does not automatically refresh results for changed analysis code/settings.

### Japanese analysis and identity evidence

`language=ja` runs Qwen Japanese transcription and the copied RomajiASR model
independently for each singer. RomajiASR phonemes are grouped into mora using the
Vocal2Midi rules; TiFA aligns that pronunciation stream. The explicit local
kashi G2P bundle supplies untimed reading candidates for the Qwen surface text.
These candidates are retained for review, rather than forced onto the aligned
pronunciation or copied from teacher to student.

Japanese analysis words carry `language=ja`, `unit_kind=mora`,
`identity_basis=phonetic_mora`, `romaji`, `kana`, `chunk_index`, and aligned
phonemes. Chunks preserve Qwen `text`, `alignment_text`, `romaji_tokens`,
`mora_tokens`, `semantic_readings`, and `semantic_reading_basis`. The Japanese
models and conversion code participate in cache fingerprints. Missing local
resources fail explicitly. Language changes and pre-integration Japanese
analyses cannot be reused through `--skip-analysis`.

The planner retains Japanese lexical/mora boundaries and never supplies Chinese
pinyin for Japanese kanji. Its pronunciation signatures distinguish contracted
kana, gemination, long vowels, and syllabic N. A phonetic mora match is a
`homophone` with `phonetic_only=true`, even when its romaji text is identical.
It contributes to usable acoustic coverage but not `semantic_match_rate` or
reliable exact-lexical timing. `phonetic_only_match_count` keeps the job
unresolved. Untimed Qwen text does not resolve kanji homophone lyric errors.
No Japanese audio or perceptual validation is claimed for this integration.

## Correspondence and uncertainty

`inference/alignment.build_alignment(teacher, student, ...)` normalizes lyric units and performs global edit alignment. Exact text costs least, matching pronunciation costs slightly more, and substitutions/deletions/insertions cost more. Chinese phonetic signatures use the aligned TiFA phonemes with tone digits removed; an installed lightweight pinyin reader can supply a fallback. Relative progress is a weak tie-breaker, rather than a pitch similarity score.

Forward/backward lexical alignment costs reveal equally plausible identities in repeated lyrics. If different repetitions support equivalent correspondence, the plan records candidate indices and an ambiguity reason. These matches become `unresolved` with quality capped at 0.25. This conservative downgrade can reject an entire indistinguishable chorus; the report states when all timed teacher lyric material would be muted.

Matches have `matched`, `homophone`, `substitution`, `missing`, `extra`, or `unresolved` status. A substitution preserves available student content while reporting a possible lyric/ASR difference. It does not convert one spoken vowel or word into another. A missing unit means that no corresponding student lyric unit was established; absent source content cannot be created by time warping.

When one track has no recognized lyric units but observed vocal activity, the strict plan treats correspondence as unresolved rather than asserting a whole song was omitted or inserted. By default, no activity-only synthesis is allowed. The explicit `--allow-acoustic-fallback` option permits low-confidence activity timing only when teacher and student voiced runs can be paired one-to-one, their relative progress is compatible, and each duration ratio stays within the configured bound. A fallback never spans an intervening pitch-unvoiced interval; incompatible or unmatched runs remain unresolved. It also permits selected ambiguous repeat mappings. Such regions remain explicitly labeled and receive no semantic-success credit. Activity-only timing does not reconstruct internal word boundaries or resolve a missing/repeated lyric sequence.

Multi-character or multi-unit ASR words without internal alignment are split into estimated equal spans and receive reduced confidence plus a warning. Precise editing requires phoneme/character-level aligned boundaries. Overlapping word boundaries are clipped to preserve ordering and their confidence is reduced; severe damage can make a region unsupported.

## Region and time-map contract

The plan uses teacher time as the output clock. Its sparse `time_map` maps **teacher seconds to student seconds**, and requires a region mask. A continuous global warp cannot express both deleted source words and missing output material safely.

Every region contains:

```python
{
    "teacher_start": 0.52,
    "teacher_end": 0.70,
    "student_start": 0.74,
    "student_end": 0.90,
    "status": "matched",
    "mode": "synthesize",
    "quality": 1.0,
    "unsupported": False,
    "teacher_anchors": [0.52, 0.55, 0.70],
    "student_anchors": [0.74, 0.77, 0.90],
    "anchor_strategy": "vowel_weighted",
    "stretch_ratio": 1.125
}
```

Valid local anchor arrays are finite and strictly increasing on both clocks. `map_times()` combines those local maps with a bounded nondecreasing diagnostic global map. The renderer interpolates only inside each allowed region, so source insertions are cut rather than stretched across. Global-map discontinuities can appear at a source deletion; consumers should use the local arrays and `mode` rather than bridge those jumps.

`synthesize` regions are semantically supported for editing, subject to quality
and structural checks. `mute` regions represent unsupported correspondence.
`acoustic_fallback` represents explicitly opted-in low-confidence activity
timing. Pitch-unvoiced gaps retain the legacy rendering mode `silence`, but are
explicitly annotated with `status="unvoiced_gap"`,
`reason="pitch_unvoiced_not_acoustic_silence"`,
`acoustic_classification="unknown"`, and `silence_confirmed=false`.
`gap_source_start` and `gap_source_end` preserve the original source-gap
candidate even when the legacy source anchors collapse to one cursor.
Their quality value is a structural pitch-gap score, not acoustic-silence or
semantic confidence.

Gap planning splits each interval into contiguous voiced runs and intervening
pitch-unvoiced intervals, so sparse RMVPE frames cannot be reported as one long
continuous missing-vocal span. This annotation refinement alone preserves the
original modes, anchors, and semantic support. The acoustic renderer separately
decides which safe interword intervals can retain source audio; a legacy
`silence` mode does not by itself establish that the waveform should be zero.

Teacher word edges are nominal target anchors. When source phoneme boundaries are available, most length change is assigned to vowel nuclei, with consonants kept near their source duration and a vowel reserve under compression. Matching vowel identities can use the teacher's relative nucleus rhythm. A word can span several target notes: teacher F0 remains continuous and is not quantized into one note per word. Without usable vowel boundaries, the region explicitly uses a linear word map. ASR/TiFA edges are estimates and can cut through continuous articulation; the renderer may refine actual source sampling inside bounded neighboring phone windows while preserving these original annotations.

The planner reports whole-word and local phoneme stretch extremes. Whole-word ratios outside the configured maximum (default 4x in either direction) are unsupported. A mild total ratio can still hide extreme local compression/stretch, so local ratios also produce quality warnings.

## Feature editing and synthesis contract

The PC-NSF-HiFiGAN bundle requires 44.1 kHz mono audio, 128 mel bins, a 512-sample hop, and 2048-sample FFT/window. Extraction uses magnitude STFT, a periodic Hann window, Slaney-normalized mel filters from 40 Hz to 16 kHz, and natural-log amplitudes with a configurable positive floor. Reflection padding follows the upstream vocoder contract. The final partial hop is padded and the synthesized waveform is trimmed to the requested exact sample count.

Mel frame centers are `(frame_index + 0.5) * 512 / 44100` seconds. The renderer converts mapped source seconds to this coordinate before interpolating mel features. It samples log-F0 inside separate voiced runs and never interpolates across an unvoiced gap.

| Render mode | Output clock | Spectral features | Pitch control |
| --- | --- | --- | --- |
| `identity` | Student | Student mel | Student F0 |
| `pitch_only` | Student | Student mel | Mapped teacher F0 in supported correspondences; student pitch elsewhere |
| `time_only` | Teacher | Region-warped student mel | Region-warped student F0 |
| `full` | Teacher | Region-warped student mel | Pitch per the three options below |

`full` and `pitch_only` take three independent options. `change_timing` selects the output clock: with it the student is warped onto the teacher's timeline, without it the student's own timeline is kept. `pitch_mode` is a three-way switch -- `off`, `teacher` or `adaptive` -- and `map_teacher_pitch` is a separate checkbox that selects whose continuous contour is sung. They select one of four pitch policies, reported per render as `pitch_policy`:

| pitch_mode | map_teacher_pitch | policy | Pitch used inside matched regions |
| --- | --- | --- | --- |
| off | - | `none` | The student's own pitch, unchanged. |
| teacher | off | `student_contour_teacher_register` | The student's own contour, moved by one constant into the teacher's register. |
| teacher | on | `teacher_contour_teacher_register` | The teacher's contour and register exactly as analysed. |
| adaptive | off | `student_contour_student_register` | The student's own contour, with each word moved onto the teacher's note level expressed in the student's own register. |
| adaptive | on | `teacher_contour_student_register` | The teacher's contour, moved into the student's register by one constant; it already carries those note levels. |

Every level in this module is the note a singer held, read by `_sustained_cents`
rather than by a median of the whole word. On the included material the
within-word pitch span has a median of about 240 cents, so the metric mixes the
glide and the onset into the note and the result depends on the measurement
method rather than on what was sung. Validated against GAME's own note extraction,
reading the held note cuts the per-word level error from 12.4 to 5.7 cents and
raises the share of words within 25 cents from 66% to 85%.

The `teacher` and adaptive-with-contour modes move by one global constant,
`contour_shift_cents`, chosen as the target register's level minus the chosen
contour's own level. A per-word correction
is deliberately not part of that mode: it would also rewrite the notes of a student
who simply sings the song differently, and it silently changed pitch on jobs where
the user asked only for a timing change. Only `student_contour_teacher_register`
and `teacher_contour_student_register` need the measured offset;
`teacher_contour_teacher_register` targets the teacher's absolute levels directly.

`adaptive` mode is the one policy that moves words individually, and only because
asking for it is the whole point of the setting. Per region the shift is
`offset_cents - difference_cents`, which places the word's median exactly on the
teacher's note level transposed into the student's register; a student who already
sings the right melody below the teacher therefore needs no correction at all. A
shift beyond `_MAXIMUM_NOTE_SHIFT_CENTS` (a tritone) is refused as a likely
octave-tracking error and a residual below `_NEGLIGIBLE_NOTE_SHIFT_CENTS` (ten
cents) is left alone; both are recorded per region with their reason in
`regions[*].note_shift_reason`. `frame_note_shifts` turns the per-region constants
into one shift per output frame, reading the region intervals on whichever clock the
output uses, holding each word's shift constant so the student's own slides and
vibrato keep their shape, and ramping over 30 ms where two words meet so the F0 the
vocoder sees does not step.

`none` is the only pitch-preserving policy, so the `identity` and `time_only`
experiments (whose pitch behavior is fixed by the mode name) and a timing-only job
with the switch off all leave the pitch exactly as sung.

`identity` and `time_only` are the pitch-preserving halves of the A/B set, so their pitch behavior is fixed by the mode name and the options do not alter them. `correction/register.py` supplies the register difference used when a move is asked for: the median log-frequency difference between the two singers over matched regions voiced on both sides, quantized to whole semitones so the rendered melody stays in a well-tempered key. Levels are robust per-region medians measured inside each singer's own interval, so one expressive word cannot dominate. The difference needs at least two comparable regions, because one word cannot separate register from a wrong note, and a measurement beyond two octaves is refused as a likely octave-tracking error. The per-region measurements are reported as evidence for the estimate; they never become per-word pitch edits.

Full/pitch-only rendering uses both singers' voicing to avoid adding a periodic source to source consonants or breath. The student's mel can remain active when F0 is zero. Different consonant and vowel boundaries can still cause a short transition where a source vowel receives zero F0; perceptual review remains necessary. Rejected or missing lyric spans are muted exactly. A synthesized loud output is attenuated before 24-bit PCM export to avoid full-scale clipping.

`correction/transitions.py` constructs continuous acoustic maps between
eligible consecutive teacher and student correspondences. The complete target
gap must consist of recognized nonlexical gap regions, and the source gap must
contain no intervening recognized word. Missing lyrics, rejected matches,
disabled fallbacks, and unmatched extra words are barriers. Original semantic
support is preserved even when a previously unresolved gap receives acoustic
audio.

For short transitions, the last inner anchor of the left word and the first
inner anchor of the right word bound a local acoustic alignment. When inner
phone anchors are unavailable, a bounded interior margin supplies the window.
Local DTW compares cepstral shape, relative energy, and spectral flatness.
When both original waveforms are supplied through the optional `teacher_audio`
and `student_audio` arguments, an additional absolute periodicity cost helps
distinguish a vowel from a low-frequency breath. This uses mean-removed 30 ms
waveform windows and overlap-normalized autocorrelation over 70--1100 Hz,
independently of RMVPE's F0/voicing classification. Energy checks exclude
inaudible windows from periodicity identity costs.

Sustained broadband-noise onsets can provide an additional timing landmark.
The mel cue requires a spectral-flatness and high-frequency-energy rise lasting
at least 20 ms. A waveform cue can also identify at least 30 ms of audible
aperiodic sound (periodicity at most 0.55) following sustained tonal material
(periodicity at least 0.75 for 20 ms). It can supply a landmark when the mel onset is absent
or substantially delayed, including low-frequency /h/ or breath. Both cue types
require energy relative to the local acoustic window, so a transition to deep
silence does not establish a noise landmark. Diagnostics distinguish
`mel_noise_rise` and `waveform_periodicity_drop` cues and record each side's
minimum duration and measured aperiodic support. Waveform audibility requires
RMS within approximately 30 dB of the window's 85th-percentile RMS. A waveform
onset replaces a later mel onset only when that onset is at least 40 ms late
and still lies inside the same sustained aperiodic run.

Each side of a noise landmark uses DTW when its shape constraints permit,
otherwise continuous linear interpolation. The diagnostic strategy name
`noise_anchored_acoustic_dtw` can therefore describe a noise-anchored map with
linear subsegments and a null cost. Its landmark kind is
`sustained_broadband_noise_onset` or `sustained_aperiodic_onset`, depending on the
selected cues. Pre-onset segment duration ratios remain between 0.25x and 4x.
Post-onset segments may use up to 6x in either direction only when the source
contains at least 60 ms of sustained audible aperiodic material; otherwise the
limit is 4x. These segment limits do not relax the total phone-window limit.
The resulting map samples a changing
source trajectory across the gap and can redistribute adjacent vowel/consonant
durations. Teacher mel features guide this timing calculation only; the
synthesized mel always comes from the student. Windows are disjoint, and their
outer source/target anchors remain fixed.

The refinement rejects a total phone-window stretch above 3x, or a target gap
above 600 ms when there is no meaningful source gap. When either real source
or target gap exceeds 800 ms, a continuous linear gap map uses a maximum 4x duration
ratio. Incompatible local DTW windows use a bounded linear phone-window map.
These structural limits prevent unlimited stretching but do not certify
phonetic identity, perceived timing, or naturalness. A nonlexical source sound
can be a breath, reverb, noise, or an unrecognized word; acoustic preservation
does not resolve that uncertainty.

`correction/durations.py` bounds each refined window with phoneme-class duration
budgets, because an unconstrained warp has a strong bias toward stretching
consonants. Local DTW cost is dominated by spectral continuity, and a stationary
fricative stretched across a teacher rest looks nearly free under that cost
while sounding like a stuck hiss. The module partitions the source window into
its phones and the rests between them and pays for the target duration in this
order: quiet rests, audible spans, vowels, and only then consonants.

A consonant may be compressed by running out of time, but its planned duration
never exceeds the length the student actually sang; the searched warp is not
used as its target. For consonants the module prefers the planner's vowel-aware
budget, which already divides the syllable between its consonant and nucleus. A
sustained resting span is classified as quiet when its own level falls more than
about 10 natural-log units below the window's phones; such spans are the most
elastic because stretching them is inaudible. Rate limits widen across three
tiers (preferred, extended, hard) so an ordinary window needs no relaxation, and
an unfittable remainder is handed out in a fixed order and reported through
`residual_unresolved` instead of being hidden inside a consonant.

Segment boundaries are pinned on both timelines and joined with a
shape-preserving monotone cubic interpolant (PCHIP). Every budgeted segment
duration is reproduced exactly, while the slope blends across the joints. That
rate blending is the warp-domain equivalent of a crossfade: a tempo step at a
phone or word joint would be audible even when source position is continuous.
One neighbouring anchor outside each window edge is added as a context knot so
the rate reaching into the window also blends. Windows with no phoneme labels
return no segments and keep the searched warp, recorded as
`duration_constraint.applied = false`.

`renderer._crossfade_splices` handles the remaining discontinuity case. Where a
sliced map left a genuine source jump at a window edge, the two sides no longer
share source material and concatenating them would click. An equal-power mel
blend over a bounded window removes that click without moving either side's
timing. Ordinary continuous joints are detected and skipped, because the
constrained mapping has already smoothed their rate. The lyric map, F0 gate and
semantic mask are untouched.

`correction/dynamics.py` prevents continuous maps from carrying a loud source
vowel through a genuinely quiet target rest. It compares actual 20 ms waveform
RMS; pitch/UV labels do not enter this calculation. For each singer separately,
the original left terminal and right initial anchor segments supply RMS
references. The greater of their 75th-percentile levels sets that singer's local
reference, excluding the intervening gap. This preserves relative quietness
without copying the teacher's absolute recording level. Insufficient flanking
phone energy leaves the gain unchanged.

A ceiling becomes active as teacher RMS falls from -20 dB to approximately
-30 dB relative to its local phone reference. It permits 6 dB of student-relative
headroom and only attenuates source material exceeding that ceiling. Comparable
low-energy consonants or breaths therefore remain audible without requiring
detected F0. Gain remains between 0.001 and 1; it never amplifies or creates a
new hard mute. The affected interval extends at most 25 ms beyond each gap edge,
clipped to the fixed outer phone anchors. Explicit missing/rejected lyrics
remain barriers even if an obsolete external transition is supplied.

RMS receives an additional 20 ms triangular smoothing kernel. Positive gain is
smoothed with a 6 ms Gaussian standard deviation over 36 ms of support, then
tapered back to unity at its outer edges. This avoids an abrupt gain drop when
the last nonzero sample exits an RMS window. The renderer interpolates the
frame gain onto sample times and applies it once after vocoder synthesis;
there is no duplicate mel attenuation. The local time map, F0, spectral source,
and semantic support are unchanged by this energy constraint. These are
relative energy safeguards, not a phonetic or perceptual correctness test.

`correction/boundaries.py` supplies bounded attack/release context at remaining
edges. Supported words that touch on the target timeline form one island;
ordinary continuous word boundaries receive no artificial fades. Up to 20 ms
of attack and 60 ms of source release may enter eligible nonlexical gaps.
With no safe source continuation, an internal terminal texture can decay for
up to 40 ms. Small target gaps allocate more room to release and avoid
overlapping contexts. Contexts overlapping a continuous acoustic transition
are removed, so the new trajectory owns its complete window.

Context mel extraction crops and reflects audio at safe lexical limits, so an
STFT window cannot borrow another word even when its center remains in range.
Real release fades over its final 35 ms; held release decays over its entire
length and follows the teacher's 20 ms RMS relative to the 15 ms pre-boundary
reference. This energy ratio can only attenuate held texture, preventing a
high-energy vowel from continuing over the teacher's quiet release. The CLI
passes teacher audio for this envelope measurement. Tapering happens in log-mel amplitude before synthesis, with a final
5 ms sample safety fade at the outside edge. This avoids the previous immediate
mel floor at word ends. Only adjacent target words with an actual source jump
of at least 25 ms get a bounded approximately 24 ms mel transition.

The target-pitch NPZ keeps original semantic `supported` separate from acoustic
`rendered` and `acoustic_context` masks. `acoustic_context` identifies rendered
frames outside semantic support; `time_map_refined` identifies complete
acoustic-refinement windows and can overlap semantic support. `source_times`
contains the actual refined map, so supported-frame source samples and sampled
student voicing can change within these windows. `acoustic_energy_gain` stores
the positive frame gain applied to the synthesized waveform. `pitch_policy`,
`register_offset_cents`, `register_offset_available`, `note_shift_applied_count`
and `timing_changed` make the exported target self-describing, so verification can
tell which clock and which register it was aiming at, and whether the words were
placed individually. `render.json` records
`options`, `timing_changed`, `pitch_policy` and `register_pitch` (the placement
kind, the applied and raw offset, the comparable-region count, the per-region
shift list, and how many words were measured and how many moved) in addition to
`acoustic_transitions` with window/gap bounds, region ownership, alignment
strategy, optional acoustic landmarks, and changed frame counts, alongside
edge contexts, source edits, unresolved counts, and rendered coverage.
Each transition also carries `duration_constraint`: applied status, segment
kinds, planned/sung/native seconds and rates, the tier used, whether any
consonant was stretched, and any residual with its `residual_unresolved` flag.
`acoustic_splice_crossfades` lists the discontinuous window edges that received
an equal-power mel blend.
When waveform features are available, `source_acoustic_mismatch` records audible
aperiodic target frames (periodicity below 0.4) mapped to audible periodic source
frames (above 0.6). This exposes remaining texture mismatches without changing
lyric identity or semantic coverage.
Its `acoustic_dynamics` entries contain the teacher/student phone-reference
intervals and levels, applied range, minimum gain, and attenuated-frame count;
`acoustic_energy_attenuated_frame_count` totals affected render frames.
Acoustic rendering does not establish missing lyric correspondence or
synthesize an absent word.

The checkpoint is loaded strictly, including legacy weight-normalization parameters. Parameter mismatches fail rather than leaving random generator weights. Source-spectrum pitch leakage and double-pitch artifacts remain possible under large corrections; they cannot be ruled out by checkpoint or duration checks.

## Verification and result interpretation

The verification worker reloads the exported audio after synthesis exits. It checks finite samples, exact rate/count, peak/clipping, and local boundary steps. RMVPE is run on the actual waveform and compared with both teacher pitch and the exported render target. When the policy targets the student's register and the exported offset is not negligible, an additional comparison against the teacher's pitch moved by that offset is reported, because agreement with the absolute teacher pitch then measures the intended transposition rather than an error. Voiced recall and missed/false voiced counts are reported alongside error on jointly voiced frames, so muting difficult material cannot silently improve a single pitch-error score. Supported and unsupported regions receive separate coverage diagnostics.

A limited independent YIN diagnostic samples stable energetic voiced windows. It helps reveal periodicity and octave errors but represents a selected subset, and can itself select a harmonic or wrong octave. The primary RMVPE comparison shares the teacher's estimator and is not independent proof of acoustic correctness.

Word/phoneme boundaries in `alignment.json` are planned edit coordinates. Default verification does not remeasure generated-word boundaries. The optional `--verify-timing` pass independently analyzes the generated waveform, matches its recognized units back to the teacher, and writes `timing.verification.json` with recognition coverage and measured onset/offset errors, including a reliable-match subset. This pass shares ASR/TiFA models with input analysis and can inherit their recognition/alignment errors; it is an additional acoustic measurement rather than ground truth. Verification does not certify intelligibility, consonant quality, timbre preservation, or a listening pass. Inspect `student.identity.wav` before judging whether correction damaged a voice beyond the baseline vocoder reconstruction.

The original baseline short run has 17 lyric units, with 16 exact matches and one suspected `了`/`啊` substitution. Planned lyric-duration coverage is 100%; the main output has exactly 322,877 samples. Its jointly voiced RMVPE error versus the teacher is 5.83 cents median / 69.61 cents p95, with teacher-voiced recall 98.92%. Optional ASR/TiFA timing remeasurement found 16 reliable exact-word correspondences, all within 20 ms for both boundaries; the differing word was excluded.

The original baseline 290.3988-second difficult run contains 477 teacher and 476 student lyric units, with 446 exact matches, 12 homophone matches, 16 suspected substitutions, three missing and two extra units. Planned lyric-duration coverage is 99.681%, but teacher-voiced semantic coverage is 93.53% and measured teacher-voiced recall is 92.10%. Jointly voiced pitch error is 6.76 cents median / 69.04 cents p95. Timing verification found 432 reliable exact-word correspondences; 427 had onset and offset errors within 50 ms. Reliable-subset mean absolute onset/offset errors were 5.59/6.68 ms. Five outliers include suspected alignment silence absorption and ASR omissions, so those scores are not manual ground truth. The refined plan contains 141 unsupported regions, including 137 untranscribed voiced runs totaling 11.215 seconds after clipping to lyric boundaries. These are mostly short voiced runs separated by pitch-unvoiced intervals that may contain audible aperiodic material; long untranscribed spans must be inspected, and lyric coverage alone overstates full-recording completeness. These results are tied to each run's artifact provenance and have no subjective listening approval.

## Attribution and asset boundaries

Vocal2Midi infrastructure keeps its Apache-2.0 license and upstream notices in `licenses/`. The adapted generator source retains an Apache-2.0 license beside the code. The bundled OpenVPI PC-NSF weights are separately licensed CC BY-NC-SA 4.0; their full English/Chinese notices and statements stay with the model. Other model weights, dictionaries, bundled native code, Python packages, and FFmpeg keep their own terms. Source-code licensing does not supersede model-weight or binary licenses.

`settings/resource_manifest.json` records copied resource hashes and verifies no reparse points/hard links. `settings/synthesis_runtime_manifest.json` records the physically copied PyTorch stack. Original source locations in those manifests identify provenance only. The migration scripts are not required during a normal job.
