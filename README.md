# CopyVocal

CopyVocal retimes a student's isolated vocal recording to a teacher recording and drives resynthesis with the teacher's continuous pitch. It uses the student's spectral features to preserve as much of the student's voice as the vocoder permits. This is an experimental command-line pipeline for Windows; perceptual voice preservation and exact rendered word boundaries still require review.

The pipeline combines Qwen3-ASR lyric recognition, TiFA forced alignment, RMVPE continuous pitch, optional GAME note extraction, and PC-NSF-HiFiGAN synthesis. Words are matched independently of their pitch, so a wrong note does not prevent lyric correspondence. Missing or ambiguous source material is reported explicitly.

## Run the included examples

The working directory already contains a physically copied Python runtime, model bundles, native dependencies, and FFmpeg. Normal inference uses the files inside this project and does not require `D:\Vocal2Midi`, the original `Z:` drive, a Conda environment, or model downloads. Windows GPU drivers remain machine dependencies.

From PowerShell in the project directory:

```powershell
# Short pair: render the corrected vocal and three controlled A/B experiments.
.\run.bat --teacher ".\Teacher.wav" --student ".\Student.wav" --output ".\outputs\short" --experiments

# Difficult long pair: use a separate output directory.
.\run.bat --teacher ".\teacher_long_hard.wav" --student ".\student_long_hard.wav" --output ".\outputs\long_hard"

# Inspect all options.
.\run.bat --help
```

Open `outputs\short\report.html` in a browser to compare the reference, original student, and generated files. The main output is `student.corrected.wav`. Audio is exported as mono 44.1 kHz, 24-bit PCM. A full correction has exactly the teacher's sample count after resampling; identity and pitch-only experiments retain the student's sample count.

The default analysis backend is DirectML (`--onnx-device dml`). The vocoder selects CUDA when available and otherwise uses CPU (`--vocoder-device auto`). For a CPU run:

```powershell
.\run.bat --onnx-device cpu --vocoder-device cpu --no-game --output ".\outputs\short_cpu"
```

CPU lyric recognition and neural synthesis can be slow. `--no-game` skips optional MIDI note analysis; continuous RMVPE F0 still drives pitch correction. The CLI and desktop language selector support `zh`, `en`, and `ja`. Inputs should be isolated, predominantly monophonic vocals; accompaniment, overlapping singers, and strong effects can confuse the analysis.

For Japanese recordings, select `ja` in the desktop interface or use:

```powershell
.\run.bat --teacher "D:\songs\teacher.wav" --student "D:\songs\student.wav" --language ja --output ".\outputs\japanese"
```

Japanese analysis preserves independent Qwen Japanese text and RomajiASR
pronunciation evidence for each singer. TiFA aligns the recognized pronunciation
at mora boundaries; local kashi G2P retains reading candidates for the Qwen text.
The pipeline keeps contracted kana, gemination, long vowels, and syllabic N
distinct, and does not interpret Japanese kanji through Chinese pinyin.
`models/romajiASR` and `models/kashi-g2p-onnx` are physical local copies.

Pronunciation matches alone cannot establish Japanese lexical identity, including
kanji homophones. Such regions are marked `phonetic_only` and keep the job
`completed_with_unresolved`; they are excluded from reliable lexical timing
scores. Raw Qwen text and untimed reading candidates remain in the analysis for
review. Japanese audio validation has not been performed. Changing language
requires a new analysis; `--skip-analysis` rejects another language's artifacts.

The three correction options are independent. `--no-change-timing` keeps the
student's own timeline and corrects pitch only. `--pitch-mode` is a three-way
switch and defaults to `off`, so the default corrects timing and leaves the pitch
exactly as the student sang it. `--map-teacher-pitch` is a separate checkbox and is
off by default.

`--pitch-mode off` moves no pitch frame at all. `--pitch-mode teacher` moves the
result into the teacher's register at the teacher's own absolute levels, using the
measured register difference as one global constant -- never a per-word
correction. `--pitch-mode adaptive` puts each word on the teacher's note level but
keeps the melody as a whole in the student's own register, so a student who sings
the song a fifth lower is corrected inside their own range instead of being forced
up to the teacher's absolute pitch.

The register difference is the median level difference between the two singers over
matched words, quantized to whole semitones so the melody stays well tempered. It
needs at least two corroborating words and is refused beyond two octaves, where an
octave-tracking error is the likelier explanation.

`--map-teacher-pitch` decides whose pitch movement is sung. In `teacher` mode it
chooses the teacher's contour over the student's, both at the teacher's own level.
In `adaptive` mode, off keeps the student's phrasing and places each word on the
teacher's note level; on uses the teacher's contour, which already carries those
levels, and moves it into the student's register in one constant -- the same notes,
plus the teacher's own slides and vibrato.

The CLI reads [settings/default.json](settings/default.json), and explicit flags override the corresponding values. Use `--config ".\settings\custom.json"` to read another settings file. The bundled vocoder requires `sample_rate` to remain 44100.

## Desktop interface

A desktop interface wraps the same pipeline for users who prefer it to the command line:

```powershell
.\run_gui.bat
```

The interface is presented in Simplified Chinese; the command line, the pipeline logs and every JSON artifact
remain in English. The two are separate on purpose: the JSON is a machine-readable contract that stays stable,
while the window translates the sentences it displays. Wording lives in `gui/text.py` as data, so it can be
reviewed or changed without touching interface logic, and a pipeline warning the table does not recognize is
shown in its original English rather than hidden.

It collects the options above, shows the worker output while the job runs, and presents the finished artifacts. It adds no inference of its own: it starts `scripts\copyvocal_cli.py` in a child process, replays that output, and reads the resulting JSON. Stopping a job kills the child and the workers it owns. Qt (PySide6) is already present in the bundled runtime, so no installation step is needed.

The interface is deliberately explicit about the pipeline's limits:

- A job that finishes with unresolved regions is reported as **unresolved**, not as success, and the regions and warnings are listed.
- Planned coverage and measured waveform quality are shown as separate values, and neither is presented as a listening-quality pass.
- `--skip-analysis` can be selected, but it does not rerun analysis for changed analysis settings or algorithms; the interface says so before starting.
- Reusing analysis, and every other option, is written to `gui_settings.json` inside the chosen output directory, so the exact settings of a run stay next to its artifacts. The equivalent `run.bat` command line is shown before starting and can be copied.

Reopening the interface on an existing output directory reviews that job without running anything. The **Results** tab lists the metrics, the per-unit correspondences and the warnings; audio is played through Qt Multimedia, and `report.html` opens in the browser from there. The corrected vocal remains `student.corrected.wav`, unchanged.

Advanced settings expose the CLI's analysis knobs with the same defaults and ranges. Window state and the last used paths are stored in `settings\gui_prefs.json`, which is not read by the pipeline.

| Option | Default | Purpose |
| --- | --- | --- |
| `--change-timing` | on | Warp the student's timeline onto the teacher's word and note boundaries. Off keeps the student's own timing, so the output is as long as the student recording rather than the teacher's; verification and the report follow whichever timeline was rendered. |
| `--pitch-mode` | `off` | Which pitch the result carries. `off` leaves the student's pitch exactly as sung, frame for frame. `teacher` moves the result into the **teacher's register** at the teacher's absolute levels; this is one global move of the whole melody, never a per-word correction. `adaptive` places each word on the **teacher's note level** while the melody as a whole stays in the **student's own register**, so a student who sings lower is corrected without being dragged out of their range. |
| `--map-teacher-pitch` | off | Use the **teacher's continuous pitch**, so slides, vibrato and note choices are the teacher's. Only applies to `--pitch-mode teacher`; `off` and `adaptive` always keep the student's own pitch movement. |

The pitch switch and the contour checkbox are independent, and the checkbox only
applies to `teacher` mode. With the switch `off` -- the default -- the pitch is left
exactly as the student sang it, so asking only for a timing change cannot alter the
pitch. A student who simply sings a phrase differently keeps that choice; the
correction only moves the pitch when it is asked for.

Adaptive placement is the one setting that moves words individually, and it exists
precisely to do that. Per word, the shift is the register offset minus that word's
own measured difference, so a word already in tune is not moved at all and a word
that is off lands exactly on the teacher's note transposed into the student's
register. A shift beyond a tritone is refused as a likely octave error and a
residual below ten cents is left alone; both are reported per word. Each word's
shift is held constant across the word, so the student's own slides and vibrato keep
their shape, and the change between two words is ramped over 30 ms instead of
stepping.
| `--max-chunk-seconds` | `18` | Bound each analysis chunk, including context. |
| `--context-seconds` | `0.75` | Preserve context around disjoint chunk cores. |
| `--minimum-alignment-quality` | `0.35` | Reject low-confidence correspondence for normal synthesis. |
| `--max-stretch` | `4` | Reject whole-word duration ratios beyond 4x or below 0.25x; the JSON key is `maximum_stretch`. |
| `--log-floor` | `0.00001` | Set the positive amplitude floor before the vocoder's natural-log mel conversion. |

## Stages, experiments, and files

| Stage | Work | Main artifacts in the selected output directory |
| --- | --- | --- |
| Analysis | Load and safely downmix audio, resample, slice with context, recognize lyrics, align phonemes, extract F0 and optional notes | `teacher.analysis.json`, `student.analysis.json`, corresponding `.TextGrid` files |
| Correspondence | Match lyric units, identify insertions/deletions and uncertainty, build local teacher-to-student timing maps | `alignment.json` |
| Synthesis | Warp student mel features, retain eligible acoustic transitions, apply continuous target F0 with voicing checks | `student.corrected.wav`, `target_pitch.npz`, `render.json` |
| Verification | Remeasure the exported waveform, check length and signal integrity, compare pitch and voiced coverage, run a limited independent YIN check | `verification.json`, `verification.measured_pitch.json` |
| Optional timing verification | Recognize and force-align the generated waveform, then compare measured word boundaries against teacher boundaries | `timing.verification.json` |
| Review | Gather audio players, correspondences, warnings, diagnostics, and provenance | `report.html`, `job.json`, `processing.log` |

`--experiments` also writes `student.identity.wav`, `student.pitch_only.wav`, and `student.time_only.wav`, plus each experiment's target-pitch arrays. Identity means neural reconstruction using the student's own mel and F0, rather than a copy of the input waveform. Time-only uses the teacher clock with the student's warped pitch. Pitch-only uses the student clock and retains the student's pitch outside supported correspondences.

The report distinguishes planned lyric coverage from measured waveform quality. Exact output length does not prove exact word timing, and pitch agreement alone does not certify timbre or consonant quality.

Timing correction preserves continuous source audio between eligible consecutive
lyric correspondences. Local acoustic alignment can redistribute the neighboring
phone trajectories across an interword gap, preserving consonants, breaths, and
other aperiodic sounds. Direct waveform periodicity can locate a sustained
vowel-to-noise change even when a low-frequency breath has no bright mel-spectrum
onset. RMVPE's unvoiced label means that no periodic F0 was
detected; it does not establish acoustic silence. The teacher's mel features
guide timing only, while the student supplies every rendered spectral frame.
Missing/rejected lyrics and unmatched source words remain barriers.

Continuous mapping also respects real acoustic rests. Each singer's original
neighboring phones establish a separate local RMS reference, so different
recording levels do not cause global attenuation. Only when the teacher becomes
relatively quiet does a smooth energy ceiling reduce an overstrong mapped
student tail. It never boosts the signal or substitutes an unvoiced label for
measured silence. The positive gain is applied once after synthesis, within the
gap and at most 25 ms of neighboring context bounded by the fixed phone anchors.

Other phrase edges retain bounded context: up to 20 ms of attack, 60 ms of real
source release, or 40 ms of decaying terminal texture constrained by the
teacher's relative release energy. Original lyric anchors and semantic coverage
remain unchanged, while the actual source map can change inside adjacent phone
windows.

Consonants are duration-bounded: a fricative or plosive is never stretched past
the length the student actually sang, because stretching stationary noise sounds
like a stuck hiss while a stretched vowel is unremarkable. Quiet rests absorb
timing first, then audible spans, then vowels; an unfittable remainder is
reported rather than hidden inside a consonant. Segment boundaries are joined by
a shape-preserving monotone cubic so the tempo is blended across joints instead
of stepping at them, and a genuine source discontinuity at a window edge gets an
equal-power mel crossfade.

`render.json` records the acoustic transitions, edge contexts,
`acoustic_dynamics` energy-reference diagnostics, each transition's
`duration_constraint` budget, and `acoustic_splice_crossfades`;
`target_pitch.npz` separates `supported`, `rendered`, `acoustic_context`, and
`time_map_refined` frame masks and exports the continuous `acoustic_energy_gain`.
These changes still require waveform and listening
review; continuity alone does not prove correct consonant timing.

Jobs with unsupported regions, acoustic fallback, ambiguous matches, substitutions, or extreme local stretch finish as `completed_with_unresolved`. Optional timing verification can also retain that status when not every teacher unit meets its timing criterion. This status requires inspection of the report rather than implying that every word was corrected successfully.

## Resume and reuse analysis

Analysis caches are stored under `cache\`. Cache keys include input content, model/code fingerprints, and analysis configuration. Completed chunk results are reused on a normal rerun, while interrupted or failed chunks can be retried. Pitch has a separate cache so lyric retries do not unnecessarily repeat RMVPE work.

```powershell
# Produce inspection artifacts without rendering.
.\run.bat --analyze-only --output ".\outputs\short"

# Render again using analysis JSON already present in this output directory.
.\run.bat --skip-analysis --experiments --output ".\outputs\short"

# Render again and optionally remeasure generated-word boundaries with ASR/TiFA.
.\run.bat --skip-analysis --verify-timing --output ".\outputs\short"
```

`--skip-analysis` verifies that the analysis belongs to the current input file hashes. It does not rerun analysis to pick up changed analysis settings or algorithms. After those changes, use a normal run. Keep the input files available: rendering reloads their waveforms. After moving the project or inputs, pass the current `--teacher` and `--student` paths with `--skip-analysis`; matching hashes allow the stored absolute source paths to be rebound while preserving their previous values for provenance. Press `Ctrl+C` to cancel; completed analysis caches remain available.

`--verify-timing` adds a separate ASR/TiFA pass over the exported waveform. It reports recognition coverage and onset/offset errors, including reliable matches separately. It can be slow and is optional; it does not turn planned boundaries into acoustic ground truth. `--skip-analysis` still runs synthesis, so this command is not a verification-only shortcut.

## What correction can and cannot do

| Case | Current behavior |
| --- | --- |
| Wrong pitch, slides, vibrato, or several notes on one syllable | Use the teacher's continuous F0 on supported voiced material, transposed into the student's own register; no note quantization is required. |
| Early/late entrance or a different syllable duration | Follow teacher word boundaries and use phoneme-aware local warps, allocating most stretch to vowels while protecting consonants. |
| Different recognized words | Mark a suspected lyric mismatch or ASR error. Available student material can be retimed and repitched, but its phonetic content is not rewritten into the teacher's word. |
| Missing student material | Mute unsupported teacher spans and report them; the system cannot create missing sung words from absent source audio. |
| Extra student words | Discard unmatched source spans instead of stretching across them. |
| Repeated lyrics or an omitted chorus | Use global text/phonetic context and a weak progress prior. Equally plausible correspondences are flagged as ambiguous and downgraded. |
| Failed ASR or untranscribed gaps | Keep lyric correspondence unresolved. Eligible consecutive correspondences can preserve their intervening source acoustics without semantic-success credit. `--allow-acoustic-fallback` explicitly enables low-confidence activity-only timing. |
| Extreme stretching or low-quality boundaries | Report the affected region and mute unsupported mappings; extreme local phoneme stretching also produces a warning. |

Each matched word is moved by the measured difference between the teacher's level
and the student's own level in that word, so a note the student got wrong is
corrected while the rest of the performance is left alone. The register offset
needs at least two matched regions with voiced pitch on both sides: a single word
cannot distinguish "this singer is a fifth below throughout" from "this one word was
sung low", and treating the difference as a register would cancel the very
correction that word needs. A per-word difference larger than an octave is clamped,
and a measured register larger than two octaves is refused as a likely octave-tracking
error. `render.json` records the offset (raw and applied), the number of words moved,
and the clamp count; `target_pitch.npz` records the pitch policy and offset, and
verification adds a register-matched pitch comparison because agreement with the
teacher's absolute pitch then measures the intended transposition rather than error.

Large pitch changes can retain traces of the original pitch or damage voice quality because the student's mel spectrum still contains source pitch information. Clean reconstruction is therefore a prerequisite for judging the correction. The included A/B files and objective measurements support that review; listening assessment has not been certified.

## Current measured examples

Pitch is mapped relative to the student's own register by default. On the
difficult pair the student sings 445.9 cents below the teacher, so that mapping
exports a -500-cent register offset and leaves the corrected voice at 185 Hz
instead of forcing it to the teacher's 247 Hz; accuracy against the aimed target
is unchanged (6.75 cents median, versus 6.79 before). See
[docs/validation.md](docs/validation.md) for that comparison and its limits.
The measurements below describe the original baseline under `outputs/short`
and `outputs/long_hard`. The earlier tail-only revision is preserved under
`outputs/short_natural` and `outputs/long_hard_natural`. Continuous acoustic
mapping uses separate `outputs/short_continuous` and
`outputs/long_hard_continuous` directories. See
[docs/validation.md](docs/validation.md) for revision-specific results and the
limitations found in the earlier approach.

The included short pair has completed an end-to-end run. Both tracks contain 17 aligned lyric units: 16 exact matches and one suspected substitution, teacher `了` versus student `啊`. Planned lyric-duration coverage is 100%; small untranscribed voiced gaps remain explicitly unsupported.

The exported correction has exactly 322,877 samples at 44.1 kHz, matching the resampled teacher. Remeasured output-versus-teacher RMVPE pitch error is 5.83 cents median and 69.61 cents at the 95th percentile on jointly voiced frames; teacher-voiced recall is 98.92%. Optional ASR/TiFA timing verification recovered 16 reliable exact-word correspondences out of 17 teacher units; the `了`/`啊` difference was excluded. All 16 reliable units had onset and offset errors within 20 ms.

The difficult long pair also completed analysis, rendering, and waveform verification. Its output follows the teacher's 290.3988-second clock. Alignment found 477 teacher units and 476 student units: 446 exact matches, 12 homophone matches, 16 suspected substitutions, three unmatched teacher units, and two unmatched student units. Planned lyric-duration coverage is 99.681%, while teacher-voiced semantic coverage is 93.53% and measured output teacher-voiced recall is 92.10%. Jointly voiced RMVPE error is 6.76 cents median / 69.04 cents p95. Timing verification found 432 reliable exact-word correspondences out of 477 teacher units; 427 had both boundaries within 50 ms. Mean absolute onset/offset errors on the reliable subset were 5.59/6.68 ms. The five outliers include suspected TiFA silence absorption and ASR omissions. The refined plan has 141 unsupported regions, including 137 untranscribed voiced runs totaling 11.215 seconds after clipping to lyric boundaries. These are mostly short voiced runs separated by pitch-unvoiced intervals, which can still contain audible aperiodic material. Long untranscribed spans require review, and high lyric coverage does not establish complete correction. Gap segmentation can increase the reported region count without changing the supported synthesis mask.

RMVPE also estimates teacher pitch, so these pitch scores are agreement measurements, not independent proof. Verification includes a separate YIN diagnostic on selected stable voiced windows. Timing measurements share ASR/TiFA models with input analysis and are not manually labeled ground truth. Default verification does not remeasure rendered word boundaries; use the optional timing pass to inspect them. Neither example has a certified listening-quality pass.

## Project layout and environment

```text
CopyVocal/
  app_gui.py         Desktop interface entry point
  application/       Job orchestration, provenance, and HTML reports
  gui/               Desktop interface over the command-line pipeline
  inference/
    API/              Vendored model adapters
    TiFA/             Forced-alignment runtime and dictionaries interface
    alignment/        Lyric correspondence and local time maps
    correction/       Region-aware feature and F0 editing
    slicer/           Bounded chunks, context, and channel checks
    vocoder/          PC-NSF generator, feature contract, and chunked synthesis
    analysis.py       Cached dual-track analysis
    verification.py   Exported-waveform measurements
  models/             Physical copies of all inference model bundles
  python/             Isolated portable Python and installed libraries
  settings/           Resource/runtime manifests and settings
  scripts/            CLI, worker processes, migration, probes, and smoke checks
  tests/              CPU behavioral and contract tests
  docs/               Architecture documentation
  licenses/           Retained Vocal2Midi licenses and notices
  cache/              Reusable analysis intermediates
  outputs/            Per-job artifacts
  ffmpeg.exe
  run.bat
  run_gui.bat
```

`python\python312._pth` isolates imports from machine and user Python packages. `settings\resource_manifest.json` records model/native resource hashes and physical-copy verification; `settings\synthesis_runtime_manifest.json` records the copied PyTorch distribution and dependencies. No symlinks, junctions, or hard links are used for the model/runtime migration.

The normal user does not need an installation step. `scripts\bootstrap.py` and `scripts\bootstrap_torch.py` are migration utilities for rebuilding the local bundles from supplied sources, not startup dependencies. A full bootstrap overlays vendored inference files and the runtime; preserve CopyVocal modifications and the isolated `_pth` configuration when rebuilding. A read-only resource audit can be run with:

```powershell
.\python\python.exe -X utf8 .\scripts\bootstrap.py --verify-only
```

Run the behavioral and contract suite with:

```powershell
.\python\python.exe -m unittest discover -s tests -p "test_*.py"
```

Three standalone probes check what unit tests cannot, and are not part of the suite above:

```powershell
# Drive one real job through the window's own code path.
.\python\python.exe -X utf8 scripts\gui_e2e_probe.py

# Confirm that cancelling stops the job and the workers it spawned.
.\python\python.exe -X utf8 scripts\gui_cancel_probe.py

# Render the window to PNG for layout review without a display.
.\python\python.exe -X utf8 scripts\gui_screenshot.py
```

See [docs/architecture.md](docs/architecture.md) for time coordinates, data contracts, memory handling, and verification limits.
See [docs/resources.md](docs/resources.md) for the copied model/runtime inventory and migration audit.
See [docs/validation.md](docs/validation.md) for the final checks, example measurements, and their limits.

## Attribution and licensing

CopyVocal reuses infrastructure from [Vocal2Midi](licenses/Vocal2Midi-NOTICE), whose source is Apache-2.0; its retained [license](licenses/Vocal2Midi-LICENSE) and [acknowledgements](licenses/Vocal2Midi-ACKNOWLEDGEMENTS.md) apply to that material. The adapted PC-NSF generator implementation has its own retained [Apache-2.0 license](inference/vocoder/LICENSE).

**The bundled PC-NSF-HiFiGAN model weights have different terms: CC BY-NC-SA 4.0.** Preserve the OpenVPI attribution and the full [vocoder notice](models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/NOTICE.txt). The source-code license does not remove the weights' noncommercial/share-alike conditions. Other copied model weights, native binaries, dictionaries, data, and runtime libraries retain their respective upstream licenses; an Apache-2.0 source notice is not a blanket license for every bundled asset. Refer to the retained notices and original model releases when using or redistributing them.
