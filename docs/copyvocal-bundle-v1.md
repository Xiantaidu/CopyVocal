# CopyVocal File (`.cvf`) — format v1

A `.cvf` file is the versioned interchange document a CopyVocal job can export. It
carries the student reference audio, the student's original F0, CopyVocal's
corrected target F0 mapped back onto source time, the notes, the region
correspondence, and a source→output time map, so an editing application can
continue from CopyVocal's correction instead of only replaying
`student.corrected.wav`.

- File extension: `.cvf` (CopyVocal File)
- Media type: `application/vnd.copyvocal.file+json;version=1`
- Text encoding: UTF-8, no BOM
- Numbers: JSON numbers; `NaN` / `Infinity` / `-Infinity` are forbidden
- Byte order: every binary array is **little-endian**
- Producer: CopyVocal `application.copyvocal_file`

---

## 1. Design goals

1. **Editable, not just playable.** A consumer receives notes, a pitch curve and
   a time map — the material an editor needs — rather than a flattened waveform.
2. **No dependency on CopyVocal internals.** The file is self-contained. A
   consumer must not need `alignment.json`, `target_pitch.npz` or any other
   implementation artifact, so CopyVocal can restructure those freely.
3. **Explicit clocks.** Student source time, teacher time, output time and
   project source time are named separately. No implicit assumptions.
4. **Verifiable before use.** Array lengths, frame rate, durations and time-map
   endpoints can all be checked without parsing audio.

---

## 2. The four clocks

| Name | Field | Meaning |
| --- | --- | --- |
| Student source time | `clocks.studentSeconds` | Time axis of `student.reference.wav`, the original student recording |
| Teacher time | `clocks.teacherSeconds` | Time axis of `teacher.reference.wav` |
| Output time | `clocks.outputSeconds` | Time axis of CopyVocal's rendered result; equals teacher time when `timingChanged=true` |
| Project source time | `clocks.projectSourceSeconds` | Time axis a consumer's clip uses; **always equal to output time** |

These members are expressed in **project source time**:

- `f0.original` / `f0.target` frame origins
- `notes[].start` / `notes[].end`
- `timeGrid.handles[].sourceSeconds`
- `markers[].start/end` when `clock == "source"`

These keep their own native clock:

- `regions[].teacherStart/teacherEnd`, `regions[].teacherAnchors` — teacher time
- `regions[].studentStart/studentEnd`, `regions[].studentAnchors` — student source time
- `words.student` — student source time
- `words.teacher` — teacher time

> **Why does project source time equal output time?**
> A time map of the form "clip source → clip output" is only well defined when
> both sides span the same total duration. A complete CopyVocal correction has
> the teacher's length, which normally differs from the student's. A consumer
> therefore pads or trims the student reference audio to the output duration and
> treats that as the project source audio. `f0.original` / `f0.target` must be 0
> throughout the padded range. The original media files are not modified.

---

## 3. Top-level structure

```json
{
  "format": "copyvocal.file",
  "formatVersion": 1,
  "producer": { ... },
  "project": { ... },
  "clocks": { ... },
  "media": { ... },
  "f0": { ... },
  "notes": [ ... ],
  "regions": [ ... ],
  "timeGrid": { ... },
  "words": { ... },
  "markers": [ ... ],
  "diagnostics": { ... }
}
```

The `format` identifier is `"copyvocal.file"`. It is frozen for v1: it is the
discriminator readers match on. It names the format, not the file extension, and
carries no vendor prefix because the format belongs to CopyVocal.

### 3.1 `producer` (required)

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | Always `"CopyVocal"` |
| `version` | string | yes | CopyVocal version |
| `createdAt` | string | yes | ISO-8601 UTC |
| `jobStatus` | string | yes | `job.json.status` |
| `jobId` | string | no | Job identifier; an input SHA-256 combination is recommended |
| `sourceJobDir` | string | no | Original output directory, for provenance only |

### 3.2 `project` (required)

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `name` | string | yes | Project name; consumers may use it as the default document name |
| `language` | string | yes | `zh` / `en` / `ja` |
| `sampleRate` | int | yes | Always 44100 |

### 3.3 `clocks` (required)

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `studentSeconds` | number | yes | Student reference duration |
| `teacherSeconds` | number | yes | Teacher reference duration |
| `outputSeconds` | number | yes | Rendered output duration |
| `projectSourceSeconds` | number | yes | Must equal `outputSeconds` |
| `outputClock` | string | yes | `"teacher"` or `"student"` |
| `timingChanged` | bool | yes | Matches `target_pitch.npz["timing_changed"]` |
| `pitchPolicy` | string | yes | e.g. `teacher_contour_student_register` |
| `registerOffsetCents` | number | yes | Global register offset applied to target pitch; 0 means none |

### 3.4 `media` (required)

```json
"media": {
  "student": {
    "role": "student",
    "path": "student.reference.wav",
    "originalPath": "D:/songs/student.wav",
    "sha256": "...",
    "sampleRate": 44100,
    "channels": 1,
    "sampleCount": 329466,
    "durationSeconds": 7.470884353741496
  },
  "teacher": { ... },
  "corrected": { ... }
}
```

- `path` is relative to the directory containing the `.cvf`; absolute paths are
  allowed but relative ones make the export directory movable.
- `student` is required; `teacher` and `corrected` are optional.
- `channels` must be 1 or 2.
- `sampleCount` and `durationSeconds` must come from the actual audio file, not
  from an estimate in the analysis JSON.

### 3.5 `f0` (required)

```json
"f0": {
  "hopSize": 160,
  "sampleRate": 16000,
  "frameStepSeconds": 0.01,
  "frameCount": 733,
  "original": {
    "encoding": "base64-f32le",
    "values": "AAAAAA...",
    "voiced": "AAAB..."
  },
  "target": {
    "encoding": "base64-f32le",
    "values": "AAAAAA...",
    "voiced": "AAAB..."
  }
}
```

- `hopSize` must be 160 and `sampleRate` must be 16000 — a 10 ms grid. v1 does
  not accept any other frame rate; an exporter working from a different grid
  must resample to 160/16000 first.
- `frameCount` is the number of float32 values in `values` and the number of
  bytes in `voiced`.
- `values`: base64 of a little-endian `float32[frameCount]` in Hz; unvoiced
  frames are written as 0.
- `voiced`: base64 of a `uint8[frameCount]` of 0/1.
- Frame `i` starts at `i * hopSize / sampleRate`.
- `original`: the student's F0 (RMVPE), mapped onto project source time; 0 in
  padded ranges.
- `target`: CopyVocal's corrected F0, mapped back onto project source time; 0
  where no target exists or nothing was rendered.
- v1 requires `target[s] > 0` only where `original[s] > 0`. Consumers may treat
  target values on unvoiced frames as absent.

### 3.6 `notes` (required, may be empty)

```json
{
  "start": 0.74,
  "end": 0.90,
  "pitchHz": 392.438,
  "midi": 67.106,
  "lyric": "我",
  "source": "teacher",
  "confidence": 1.0
}
```

- Times use project source time, ascending by `start`, non-overlapping.
- `pitchHz` is the **target** pitch, already including the register offset.
- `midi` is diagnostic only; consumers should not depend on it.
- `source`: `"teacher"` / `"student"` / `"target"`, recording where the note came from.
- `lyric` may be empty. v1 consumers are not required to display it; it is
  reserved for later use.

### 3.7 `regions` (required, may be empty)

```json
{
  "teacherStart": 0.52,
  "teacherEnd": 0.70,
  "studentStart": 0.74,
  "studentEnd": 0.90,
  "status": "matched",
  "mode": "synthesize",
  "quality": 1.0,
  "unsupported": false,
  "reason": "matched",
  "teacherAnchors": [0.52, 0.55, 0.70],
  "studentAnchors": [0.74, 0.77, 0.90],
  "stretchRatio": 1.125
}
```

- `mode`: `synthesize` / `mute` / `silence` / `acoustic_fallback`.
- A region spans one word. `studentStart/End` and `teacherStart/End` are that
  word's boundaries in the two clocks.
- `teacherAnchors` / `studentAnchors` are **sub-word** boundaries (typically
  phonemes), equal length and strictly increasing. They describe the internal
  alignment for diagnostics; they are not the granularity of `timeGrid` (see
  4.5).
- `mute` and `silence` regions carry a zero-length student span: there is no
  source material to map.
- Time domains per section 2.

### 3.8 `timeGrid` (required)

```json
"timeGrid": {
  "totalDurationSeconds": 7.321473922902494,
  "handles": [
    { "sourceSeconds": 0.0, "outputSeconds": 0.0, "kind": "clipStart", "confidence": "default" },
    { "sourceSeconds": 0.74, "outputSeconds": 0.52, "kind": "onsetVoiced", "confidence": "high" },
    { "sourceSeconds": 0.90, "outputSeconds": 0.70, "kind": "internalOnset", "confidence": "high" },
    { "sourceSeconds": 7.321473922902494, "outputSeconds": 7.321473922902494, "kind": "clipEnd", "confidence": "default" }
  ]
}
```

- The first handle must be `clipStart` at `(0, 0)` with `confidence = "default"`.
- The last handle must be `clipEnd` at `(D, D)` with `confidence = "default"`.
- `sourceSeconds` and `outputSeconds` must both be strictly increasing.
- `D = clocks.projectSourceSeconds = clocks.outputSeconds`.
- `kind`: `clipStart` / `clipEnd` / `onsetVoiced` / `onsetSibilant` /
  `onsetSilence` / `internalOnset` / `userAdded`.
- `confidence`: `default` / `high`. `high` marks an externally derived fact —
  the anchor's `sourceSeconds` never changes once exported. Editing
  applications may apply a minimum spacing between user-placed anchors; imported
  `high` anchors are exempt from that because they record measurement, not
  preference.
- Between two consecutive handles the mapping is linear: an output time `t`
  between `handles[i].outputSeconds` and `handles[i+1].outputSeconds` maps to the
  source time that divides the segment in the same ratio.

### 3.9 `words` (optional)

```json
"words": {
  "student": [
    { "text": "我", "start": 0.74, "end": 0.90,
      "phonemes": [{ "text": "w", "start": 0.74, "end": 0.77 }] }
  ],
  "teacher": [ ... ]
}
```

- Provenance and future lyric features. v1 consumers may ignore it.
- Student words use student source time; teacher words use teacher time.

### 3.10 `markers` (optional)

```json
"markers": [
  {
    "start": 0.51, "end": 0.52,
    "clock": "output",
    "kind": "unresolved",
    "label": "voiced_teacher_region_without_lyric_correspondence"
  }
]
```

- `clock`: `source` or `output`.
- Diagnostic only in v1; consumers may ignore it.

### 3.11 `diagnostics` (optional)

```json
"diagnostics": {
  "warnings": ["Possible lyric mismatch or ASR error: ..."],
  "unresolvedRegionCount": 1,
  "unsupportedDurationSeconds": 0.12
}
```

---

## 4. Export rules

### 4.1 Output and project source duration

```
timingChanged = target_pitch.npz["timing_changed"]
outputSeconds = render.json["experiments"]["full"]["sample_count"] / sample_rate
projectSourceSeconds = outputSeconds
```

When `timingChanged == false`, `outputSeconds = studentSeconds` and the time grid
is the identity map.

### 4.2 Original F0

Take `student.analysis.json["pitch"]` (10 ms grid) and fit it to the project
source frame count:

```
frameCount = round(projectSourceSeconds / 0.01) + 1
original[i] = student_f0[i]  (i < student_frame_count else 0)
```

If the student analysis is not on a 10 ms grid, resample to 160/16000 first.

### 4.3 Target F0

`target_pitch.npz` holds `times` (output time) and `source_times` (the student
source time each output frame corresponds to). CopyVocal's target pitch lives on
the vocoder frame grid (~11.6 ms), so scattering it onto the 10 ms grid with
`round(source_time / 0.01)` leaves a hole roughly every sixth frame, and a
consumer reading the curve sees periodic drop-outs.

Treat the target as a function of `source_times` and interpolate at each 10 ms
frame:

```
valid = rendered & finite(source_times) & finite(f0_hz) & f0_hz > 0
sort (source_times[valid], f0_hz[valid]) by source time, averaging duplicates in the log domain
for s in range(frameCount):
    t = s * 0.01
    find the bracketing valid frames t0 <= t <= t1
    if t1 - t0 <= 0.030:
        target[s] = exp(lerp(log(f0_0), log(f0_1), (t - t0) / (t1 - t0)))
    elif t is within one target frame step of the first/last valid frame:
        target[s] = value of the nearest valid frame
    else:
        target[s] = 0
```

- The `0.030` threshold bridges one missing vocoder frame (11.6 ms × 2 ≈ 23 ms)
  without filling genuine silence — breaths, consonants, unrendered regions —
  with invented pitch.
- Interpolating in the log domain avoids the bias arithmetic averaging puts on
  pitch.
- Apply a **5-frame median filter** over each contiguous voiced run. One- to
  three-frame pitch spikes sit below the vocoder's time resolution and are
  inaudible in the rendered audio, but they survive verbatim into a consumer's
  curve and surface as spurious vibrato at note joins.
- Requirement: `target[s] > 0` only where `original[s] > 0`. On unvoiced frames
  the exporter writes 0, so consumers do not end up with correction segments
  fragmented around holes.

### 4.4 Notes

Prefer `teacher.analysis.json["notes"]` (teacher GAME notes):

1. For each teacher note `[ts, te]`, take output frames with `times ∈ [ts, te]`
   and `rendered` set.
2. `sourceStart = min(source_times)`, `sourceEnd = max(source_times)`.
3. `pitchHz = median(target_f0_hz)` over frames with value > 0; if there are
   none, use `midi_to_hz(midi + registerOffsetCents/100)`.
4. Clamp to `[0, projectSourceSeconds]` and drop notes shorter than 20 ms.
5. Sort and remove overlap: a note's `start` may not precede the previous note's
   `end`.

If the teacher has no notes, fall back to student GAME notes; if those are also
absent, segment stable runs out of `target` F0.

### 4.5 TimeGrid

The grid is emitted at **word** granularity.

1. Add `(0, 0)` and `(D, D)`.
2. For every region with `studentEnd > studentStart`, add
   `(studentStart, teacherStart)` and `(studentEnd, teacherEnd)`.
3. Keep only pairs with `0 ≤ source ≤ D` and `0 ≤ output ≤ D`.
4. Sort by source; skip entries whose source or output is not increasing.
5. Mark every internal handle `confidence = "high"`, both endpoints `"default"`.
6. If no internal handle survives, emit the identity grid.

Do not expand `teacherAnchors` / `studentAnchors` into handles. Those are
sub-word splits: a 17-word sentence produces 50 phoneme boundaries against 18
word boundaries, which puts a handle on every syllable — a density nobody edits
by hand, and one that makes a consumer's editing-spacing validation fail the
whole grid. Word boundaries are what a user actually drags.

### 4.6 Media

- `student`: prefer `student.reference.wav` (already downmixed and resampled),
  otherwise `job.json.student`.
- `teacher`: prefer `teacher.reference.wav`, otherwise `job.json.teacher`.
- `corrected`: include when `student.corrected.wav` exists.
- `sha256` is optional but recommended; the exporter computes it for `student`
  and `teacher` by default.

---

## 5. Validation checklist

A consumer must verify, and an exporter should be able to guarantee:

- `format == "copyvocal.file"` and `formatVersion == 1`
- `clocks.projectSourceSeconds == clocks.outputSeconds > 0`
- `f0.hopSize == 160`, `f0.sampleRate == 16000`
- decoded `values` byte length == `frameCount * 4`
- decoded `voiced` byte length == `frameCount`
- when `clocks.timingChanged == false`, the time grid is the identity map
- time grid: endpoints correct, both axes strictly increasing, total duration
  conserved
- `notes` times inside `[0, projectSourceSeconds]`
- media paths exist and are readable, channels 1 or 2, sample rate > 0
- every number finite

---

## 6. Version policy

- `formatVersion` increments only on **incompatible** changes.
- Adding optional fields does not increment it; older readers ignore unknown
  fields.
- A reader must reject a version higher than it supports, with a clear error.
- v1 readers may ignore `words`, `markers` and `diagnostics` entirely; all three
  may be absent.

---

## 7. Minimal example

```json
{
  "format": "copyvocal.file",
  "formatVersion": 1,
  "producer": {
    "name": "CopyVocal",
    "version": "1.0",
    "createdAt": "2026-11-10T10:00:00+00:00",
    "jobStatus": "completed_with_unresolved"
  },
  "project": { "name": "demo", "language": "zh", "sampleRate": 44100 },
  "clocks": {
    "studentSeconds": 7.470884353741496,
    "teacherSeconds": 7.321473922902494,
    "outputSeconds": 7.321473922902494,
    "projectSourceSeconds": 7.321473922902494,
    "outputClock": "teacher",
    "timingChanged": true,
    "pitchPolicy": "teacher_contour_student_register",
    "registerOffsetCents": 0.0
  },
  "media": {
    "student": {
      "role": "student",
      "path": "student.reference.wav",
      "sampleRate": 44100,
      "channels": 1,
      "sampleCount": 329466,
      "durationSeconds": 7.470884353741496
    }
  },
  "f0": {
    "hopSize": 160,
    "sampleRate": 16000,
    "frameStepSeconds": 0.01,
    "frameCount": 733,
    "original": { "encoding": "base64-f32le", "values": "...", "voiced": "..." },
    "target": { "encoding": "base64-f32le", "values": "...", "voiced": "..." }
  },
  "notes": [
    { "start": 0.74, "end": 0.90, "pitchHz": 392.438, "midi": 67.106,
      "lyric": "我", "source": "teacher", "confidence": 1.0 }
  ],
  "regions": [
    { "teacherStart": 0.52, "teacherEnd": 0.70,
      "studentStart": 0.74, "studentEnd": 0.90,
      "status": "matched", "mode": "synthesize", "quality": 1.0,
      "unsupported": false, "reason": "matched",
      "teacherAnchors": [0.52, 0.55, 0.70],
      "studentAnchors": [0.74, 0.77, 0.90],
      "stretchRatio": 1.125 }
  ],
  "timeGrid": {
    "totalDurationSeconds": 7.321473922902494,
    "handles": [
      { "sourceSeconds": 0.0, "outputSeconds": 0.0, "kind": "clipStart", "confidence": "default" },
      { "sourceSeconds": 0.74, "outputSeconds": 0.52, "kind": "onsetVoiced", "confidence": "high" },
      { "sourceSeconds": 0.90, "outputSeconds": 0.70, "kind": "internalOnset", "confidence": "high" },
      { "sourceSeconds": 7.321473922902494, "outputSeconds": 7.321473922902494, "kind": "clipEnd", "confidence": "default" }
    ]
  }
}
```
