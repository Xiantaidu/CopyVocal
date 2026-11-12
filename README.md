<div align="center">

# 🎤 CopyVocal

**Retime a student's vocal to a teacher's performance — keeping the student's own voice.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?logo=windows&logoColor=white)](#-requirements)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Model%20weights-CC%20BY--NC--SA%204.0-orange.svg)](#-attribution-and-licensing)

</div>

---

CopyVocal takes an isolated student vocal and a teacher recording, then **moves the student's
timing onto the teacher's clock** while **driving pitch from the teacher's continuous F0**.
Every rendered spectral frame comes from the student, so the output sounds like the student —
corrected, not replaced.

It is an experimental research pipeline for Windows. Word boundaries and perceptual voice
quality still deserve a human listen before you ship anything.

<br>

## ✨ What it does

| | |
| --- | --- |
| 🗣️ **Lyric recognition** | Qwen3-ASR transcribes both takes |
| 🎯 **Forced alignment** | TiFA aligns phonemes to the audio |
| 🎵 **Pitch tracking** | RMVPE extracts continuous F0 with a separate voiced mask |
| 🎼 **Note extraction** | Optional GAME note analysis for MIDI-style diagnostics |
| 🌊 **Resynthesis** | PC-NSF-HiFiGAN vocoder renders from the student's mel spectrum |
| ✅ **Verification** | Remeasures the exported waveform; optional ASR/TiFA timing pass |

Words are matched **independently of their pitch**, so a wrong note never blocks lyric
correspondence. Missing or ambiguous material is always reported explicitly — never
silently papered over.

<br>

## 🚀 Quick start

The repository ships with a portable Python runtime, model bundles, native dependencies and
FFmpeg. Normal inference needs **nothing installed** and **no model downloads**.

```powershell
# Short pair — corrected vocal plus three controlled A/B experiments
.\run.bat --teacher ".\Teacher.wav" --student ".\Student.wav" --output ".\outputs\short" --experiments

# Difficult long pair
.\run.bat --teacher ".\teacher_long_hard.wav" --student ".\student_long_hard.wav" --output ".\outputs\long_hard"

# All options
.\run.bat --help
```

Open `outputs\short\report.html` to compare the reference, the original student, and the
generated files side by side.

> **Main output:** `student.corrected.wav` — mono, 44.1 kHz, 24-bit PCM.
> A full correction has exactly the teacher's sample count; identity and pitch-only
> experiments keep the student's.

**Editable OpenTune hand-off.** Add `--export-opentune` to also write
`<output>\opentune.cvf`, a versioned bundle that OpenTune imports as an editable project
(notes, original F0, target-F0 corrections and the teacher time grid) instead of a plain
audio file. Existing outputs can be exported without re-running inference:

```powershell
python -m application.opentune_bundle ".\outputs\short"
```

See `D:\OpenTune-standalone\docs\copyvocal-import.md` for the import steps and the
four-clock/zero-padding rules.

**CPU-only run** (slower, no GPU required):

```powershell
.\run.bat --onnx-device cpu --vocoder-device cpu --no-game --output ".\outputs\short_cpu"
```

The default backend is DirectML (`--onnx-device dml`). The vocoder picks CUDA when available
and falls back to CPU (`--vocoder-device auto`).

<br>

## 🎛️ The three correction switches

They are **independent** — asking for only one cannot silently change the others.

<table>
<tr><th>Option</th><th>Default</th><th>Effect</th></tr>
<tr>
<td><code>--change-timing</code></td><td>on</td>
<td>Warps the student's timeline onto the teacher's word and note boundaries. Turn it off and the output is as long as the <em>student</em> recording.</td>
</tr>
<tr>
<td><code>--pitch-mode</code></td><td><code>off</code></td>
<td><code>off</code> leaves pitch exactly as sung, frame for frame.<br><code>teacher</code> moves the whole melody into the teacher's register — one global constant, never per-word.<br><code>adaptive</code> puts each word on the teacher's note level while the melody stays in the student's own register.</td>
</tr>
<tr>
<td><code>--map-teacher-pitch</code></td><td>off</td>
<td>Use the teacher's <em>continuous contour</em>, so slides and vibrato are theirs. Applies to <code>teacher</code> mode; <code>adaptive</code> transposes that contour into the student's register instead.</td>
</tr>
</table>

`adaptive` is the only setting that moves words individually — precisely because it must. The
shift per word is the register offset minus that word's own measured difference, so an
in-tune word stays untouched and an off word lands on the teacher's note in the student's
range. Shifts beyond a tritone are refused as likely octave errors; residuals under ten cents
are left alone. Both are reported per word.

<br>

## 🌏 Languages

The CLI and desktop accept `zh`, `en`, and `ja`.

```powershell
.\run.bat --teacher "D:\songs\teacher.wav" --student "D:\songs\student.wav" --language ja --output ".\outputs\japanese"
```

Japanese analysis keeps independent Qwen text and RomajiASR pronunciation evidence per singer.
TiFA aligns at mora boundaries and local kashi G2P retains reading candidates, so contracted
kana, gemination, long vowels and syllabic N stay distinct — kanji are never read through
Chinese pinyin.

Pronunciation matches alone cannot establish lexical identity (kanji homophones). Those
regions are marked `phonetic_only`, keep the job `completed_with_unresolved`, and are excluded
from lexical timing scores. Japanese audio has not been validated end to end.

Changing language requires a fresh analysis; `--skip-analysis` rejects another language's
artifacts.

<br>

## 🖥️ Desktop interface

```powershell
.\run_gui.bat
```

A Fluent-style window over the same pipeline: icon sidebar, three pages
(**Correction**, **Results**, **Settings**), card-based controls, light and dark themes.
PySide6 is already in the bundled runtime — the widgets live in `gui/fluent_compat.py` and
the icons in `gui/assets/icons/`.

The window is bilingual. **The CLI, the pipeline logs and every JSON artifact stay in
English** — the JSON is a machine-readable contract. Only the interface text is translated,
live, through `gui/i18n.py`. A pipeline warning the table does not recognize is shown in its
original English rather than hidden.

The GUI adds **no inference of its own**. It starts `scripts\copyvocal_cli.py` in a child
process, replays its output, and reads the resulting JSON. Stopping a job kills the child and
the workers it owns.

**OpenTune hand-off in the GUI.** Enable **Also export an OpenTune bundle (.cvf)** in the
run options to write `opentune.cvf` with the job. Finished jobs can also be exported on
demand from the Results page with **Export OpenTune bundle**.

It is deliberately explicit about the pipeline's limits:

- A job with unresolved regions is reported as **unresolved**, not success — with the regions and warnings listed.
- Planned coverage and measured waveform quality are shown as **separate values**; neither is presented as a listening-quality pass.
- `--skip-analysis` is selectable, but the window says up front that it will not rerun analysis for changed settings.
- Every option is written to `gui_settings.json` in the output directory, and the equivalent `run.bat` command is shown and copyable before the run starts.

Reopening on an existing output directory **reviews** that job without running anything.
Window state, theme, language and last paths live in `settings\gui_prefs.json`, which the
pipeline never reads.

<br>

## 📂 Stages and artifacts

| Stage | Work | Main artifacts |
| --- | --- | --- |
| **Analysis** | Downmix, resample, slice with context, recognize lyrics, align phonemes, extract F0 and notes | `teacher.analysis.json`, `student.analysis.json`, `.TextGrid` |
| **Correspondence** | Match lyric units, flag insertions/deletions and uncertainty, build local time maps | `alignment.json` |
| **Synthesis** | Warp student mel features, keep eligible transitions, apply target F0 with voicing checks | `student.corrected.wav`, `target_pitch.npz`, `render.json` |
| **Verification** | Remeasure the waveform, check length and signal integrity, compare pitch and coverage, YIN cross-check | `verification.json`, `verification.measured_pitch.json` |
| **Timing check** *(optional)* | Recognize and force-align the generated waveform, compare word boundaries | `timing.verification.json` |
| **Review** | Collect players, correspondences, warnings, diagnostics, provenance | `report.html`, `job.json`, `processing.log` |

With `--experiments` you also get `student.identity.wav`, `student.pitch_only.wav` and
`student.time_only.wav`.

- **Identity** — neural reconstruction from the student's own mel and F0 (not a copy of the input).
- **Time-only** — teacher clock, student's warped pitch.
- **Pitch-only** — student clock, student's pitch outside supported correspondences.

<br>

## ⚠️ What correction can and cannot do

| Case | Behavior |
| --- | --- |
| Wrong pitch, slides, vibrato, several notes on one syllable | Uses the teacher's continuous F0 on supported voiced material, transposed into the student's register. No note quantization required. |
| Early/late entrance, different syllable duration | Follows teacher boundaries with phoneme-aware warps — most stretch goes to vowels, consonants are protected. |
| Different recognized words | Flags a suspected lyric mismatch or ASR error. Material can be retimed and repitched, but its phonetic content is **not** rewritten into the teacher's word. |
| Missing student material | Mutes unsupported teacher spans and reports them. Absent audio cannot be invented. |
| Extra student words | Discards unmatched source spans instead of stretching across them. |
| Repeated lyrics, omitted chorus | Uses global text/phonetic context plus a weak progress prior. Equally plausible matches are flagged ambiguous and downgraded. |
| Failed ASR, untranscribed gaps | Keeps correspondence unresolved. `--allow-acoustic-fallback` explicitly enables low-confidence activity-only timing. |
| Extreme stretch, low-quality boundaries | Reports the region and mutes unsupported mappings. |

**Consonants are duration-bounded.** A fricative or plosive is never stretched past what the
student actually sang — stretched noise sounds like a stuck hiss, while a stretched vowel is
unremarkable. Quiet rests absorb timing first, then audible spans, then vowels; anything
unfittable is reported rather than hidden inside a consonant.

Jobs with unsupported regions, acoustic fallback, ambiguous matches, substitutions or extreme
local stretch finish as **`completed_with_unresolved`**. That status means *read the report* —
it does not mean every word was corrected.

<br>

## ♻️ Resume and reuse

Analysis caches live under `cache\`, keyed by input content, model/code fingerprints and
analysis configuration. Completed chunks are reused; interrupted ones retry. Pitch has its own
cache, so lyric retries don't repeat RMVPE work.

```powershell
# Inspection artifacts only, no rendering
.\run.bat --analyze-only --output ".\outputs\short"

# Render again from existing analysis
.\run.bat --skip-analysis --experiments --output ".\outputs\short"

# Render again and remeasure word boundaries with ASR/TiFA
.\run.bat --skip-analysis --verify-timing --output ".\outputs\short"
```

`--skip-analysis` checks the analysis against the current input hashes. It does **not** rerun
analysis to pick up changed settings or algorithms — do a normal run after those. After moving
the project, pass the current paths and matching hashes will rebind them while keeping the
originals for provenance. `Ctrl+C` cancels; completed caches survive.

<br>

## 📊 Measured examples

> These describe the original baseline runs under `outputs/short` and `outputs/long_hard`.
> See [docs/validation.md](docs/validation.md) for revision-specific results and the limits
> found in earlier approaches.

**Short pair** — 17 aligned lyric units: 16 exact matches, one suspected substitution
(teacher `了` vs student `啊`). Planned coverage 100%.

- Output: exactly 322,877 samples @ 44.1 kHz, matching the resampled teacher
- RMVPE pitch error vs teacher: **5.83 cents** median, 69.61 cents p95
- Teacher-voiced recall: **98.92%**
- Timing check: 16/16 reliable units within **20 ms** on both boundaries

**Difficult long pair** — 477 teacher units vs 476 student units: 446 exact, 12 homophone,
16 suspected substitutions, 3+2 unmatched. Output follows the teacher's 290.3988 s clock.

- Planned coverage 99.681%; teacher-voiced semantic coverage 93.53%
- RMVPE error: 6.76 cents median / 69.04 cents p95; recall 92.10%
- Timing check: 432/477 reliable matches, 427 within 50 ms
- Mean absolute onset/offset error on the reliable subset: **5.59 / 6.68 ms**

The default register mapping is relative to the student: on the difficult pair the student
sings 445.9 cents below the teacher, so the export carries a −500-cent offset and leaves the
voice at 185 Hz instead of forcing 247 Hz — with accuracy against the aimed target unchanged
(6.75 vs 6.79 cents median).

**These are agreement measurements, not independent proof.** RMVPE estimates teacher pitch
too, timing metrics share models with input analysis, and no example has a certified
listening-quality pass.

<br>

## 📁 Project layout

```text
CopyVocal/
├── app_gui.py          Desktop entry point
├── run.bat             CLI launcher
├── run_gui.bat         GUI launcher
├── application/        Job orchestration, provenance, HTML reports
├── gui/                Desktop interface (presentation only)
├── inference/
│   ├── API/            Vendored model adapters
│   ├── TiFA/           Forced-alignment runtime and dictionaries
│   ├── alignment/      Lyric correspondence and local time maps
│   ├── correction/     Region-aware feature and F0 editing
│   ├── slicer/         Bounded chunks, context, channel checks
│   ├── vocoder/        PC-NSF generator, feature contract, chunked synthesis
│   ├── analysis.py     Cached dual-track analysis
│   └── verification.py Exported-waveform measurements
├── models/             Physical copies of inference model bundles
├── python/             Isolated portable Python and libraries
├── settings/           Runtime manifests and default settings
├── scripts/            CLI, workers, migration, probes
├── tests/              CPU behavioral and contract tests
├── docs/               Architecture, resources, validation
├── licenses/           Retained upstream licenses and notices
├── cache/              Reusable analysis intermediates
└── outputs/            Per-job artifacts
```

`python\python312._pth` isolates imports from machine and user packages.
`settings\resource_manifest.json` records model and native resource hashes; no symlinks,
junctions or hard links are used anywhere in the model or runtime layout.

<br>

## 🧪 Development

```powershell
# Behavioral and contract suite
.\python\python.exe -m unittest discover -s tests -p "test_*.py"

# Read-only resource audit
.\python\python.exe -X utf8 .\scripts\bootstrap.py --verify-only
```

Three standalone probes cover what unit tests cannot:

```powershell
# Drive one real job through the window's own code path
.\python\python.exe -X utf8 scripts\gui_e2e_probe.py

# Confirm cancellation stops the job and its workers
.\python\python.exe -X utf8 scripts\gui_cancel_probe.py

# Render the window to PNG for layout review, headless
.\python\python.exe -X utf8 scripts\gui_screenshot.py
```

The bundled runtime requires `sample_rate` to stay at **44100** in the vocoder feature
contract. `scripts\bootstrap.py` and `scripts\bootstrap_torch.py` are migration utilities for
rebuilding local bundles — not startup dependencies.

📖 Further reading: [architecture](docs/architecture.md) ·
[resources](docs/resources.md) · [validation](docs/validation.md)

<br>

## 📋 Requirements

- Windows 10 / 11 (the bundled runtime is Windows-native)
- GPU drivers for DirectML or CUDA; CPU fallback always works
- Inputs should be **isolated, predominantly monophonic vocals** — accompaniment, overlapping
  singers and heavy effects confuse the analysis

<br>

## 📜 Attribution and licensing

CopyVocal reuses infrastructure from [Vocal2Midi](licenses/Vocal2Midi-NOTICE), whose source is
Apache-2.0; its retained [license](licenses/Vocal2Midi-LICENSE) and
[acknowledgements](licenses/Vocal2Midi-ACKNOWLEDGEMENTS.md) apply to that material. The adapted
PC-NSF generator implementation carries its own retained
[Apache-2.0 license](inference/vocoder/LICENSE).

> ⚠️ **The bundled PC-NSF-HiFiGAN weights are CC BY-NC-SA 4.0 — not Apache-2.0.**
> Preserve the OpenVPI attribution and the full
> [vocoder notice](models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/NOTICE.txt). The source
> code license does not remove the weights' noncommercial and share-alike conditions.

Other copied model weights, native binaries, dictionaries, data and runtime libraries retain
their own upstream licenses. An Apache-2.0 source notice is **not** a blanket license for every
bundled asset — refer to the retained notices and the original model releases when using or
redistributing them.

<br>

<div align="center">
<sub>Built for people who want their own voice back, in tune and on time.</sub>
</div>
