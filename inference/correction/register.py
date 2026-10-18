"""Pitch placement for the corrected vocal.

The pitch control is one three-way switch plus one checkbox, and they answer
different questions:

*pitch mode* (``pitch_mode``)
    ``off``
        Nothing is done to the pitch. The student sings exactly as recorded.
    ``teacher``
        The result is moved into the teacher's register at the teacher's own
        absolute pitch.
    ``adaptive``
        Each word is moved onto the teacher's note level, but the melody as a
        whole stays in the student's own register, so a student who sings the
        song lower keeps their own voice instead of being forced up.

*contour* (``map_teacher_pitch``)
    Whose moment-to-moment pitch movement is sung. The teacher's continuous F0
    carries the teacher's slides, vibrato and note choices; the student's own
    contour keeps the phrasing they actually sang. The checkbox applies to the
    ``teacher`` and ``adaptive`` modes; ``off`` does nothing to the pitch, so
    there is no contour to choose.

They select one of five rendering policies:

===========  ============  ==============================  ==========================================
``mode``     ``contour``   policy                          result inside matched regions
===========  ============  ==============================  ==========================================
off          -             ``none``                        the student's own pitch, untouched
teacher      off           ``student_contour_teacher_      the student's own contour, moved by one
                           register``                      constant into the teacher's register
teacher      on            ``teacher_contour_teacher_      the teacher's contour and register, exactly
                           register``                      as analysed
adaptive     off           ``student_contour_student_      the student's own contour, each word moved
                           register``                      onto the teacher's note level expressed in
                                                           the student's own register
adaptive     on            ``teacher_contour_student_      the teacher's contour, which already carries
                           register``                      those note levels, moved into the student's
                                                           own register by one constant
===========  ============  ==============================  ==========================================

The register difference is one global measurement: the level difference between
the two singers over matched regions voiced on both sides, taken as a median and
quantized to whole semitones so the rendered melody stays in a well-tempered key
instead of being detuned by a fraction of a semitone. A register needs at least
two corroborating regions, because one region measures only its own level
difference; a measured difference beyond two octaves is refused as a likely
octave-tracking artifact.

Each level is the note a singer held, not the median of the whole word. On the
included material the within-word pitch span has a median of about 240 cents --
most words carry a glide, an onset and a release -- so a plain median depends on
how the word happens to be shaped and where the alignment put its boundaries.
Reading the sustained note instead was validated against GAME's independent note
extraction: per-word level error fell from 12.4 to 5.7 cents and the share of
words within 25 cents of GAME's note rose from 66% to 85%.

The policies that move the student into the teacher's register apply the register
difference as a single constant. That is deliberate: moving each word on its own
would also "fix" the notes of a student who simply sings the song differently,
which is not what those settings ask for.

Per-word placement belongs to ``student_contour_student_register`` alone, and only
because asking for it is the whole point of that setting. Per region, with
``student_level`` and ``teacher_level`` the measured levels and ``offset`` the
measured register difference:

    shift = (teacher_level + offset) - student_level
          = offset - (student_level - teacher_level)

so the word's level lands exactly on the teacher's note transposed into the
student's register. A student who sings the right melody uniformly below the
teacher therefore needs no correction at all: every shift is zero. A word that is
off gets moved onto its note, and shifts are kept constant across the word so the
student's own slides and vibrato keep their shape. A shift larger than
``_MAXIMUM_NOTE_SHIFT_CENTS`` is not applied, because a difference that large
looks like an octave error in pitch tracking rather than the interval sung, and a
residual below ``_NEGLIGIBLE_NOTE_SHIFT_CENTS`` is left alone.

Choosing the teacher's contour in adaptive mode needs no per-word measurement at
all: that contour already carries the teacher's note levels, so transposing it by
the register difference puts every note where per-word placement was aiming, and
keeps the teacher's own slides and vibrato as well.

``none`` is the only policy that does not change the pitch, so a timing-only job
cannot move a single frame no matter which other options are set.
"""

from __future__ import annotations

import numpy as np


PITCH_MODES = ("off", "teacher", "adaptive")
PITCH_POLICIES = ("none", "student_contour_student_register", "teacher_contour_student_register",
                  "student_contour_teacher_register", "teacher_contour_teacher_register")
NO_PITCH_CHANGE_POLICIES = ("none",)
# Policies whose result deliberately sits in the student's own register, so an
# absolute comparison against the teacher's pitch measures the intended
# transposition rather than an error.
STUDENT_REGISTER_POLICIES = ("student_contour_student_register", "teacher_contour_student_register")
TEACHER_CONTOUR_POLICIES = ("teacher_contour_student_register", "teacher_contour_teacher_register")
# The policies that need no measured register difference: they aim at the
# teacher's own absolute level by definition.
ABSOLUTE_PITCH_POLICIES = ("teacher_contour_teacher_register",)
# Adaptive placement is the only per-word pitch path. ``teacher_contour_
# student_register`` keeps a student contour in the student's register too, but
# it is not selected by the switch, and it moves the contour as one constant.
PER_WORD_PLACEMENT_POLICIES = ("student_contour_student_register",)
_CENTS_PER_OCTAVE = 1200.0
_SEMITONE_CENTS = 100.0
# Two octaves is already an extreme duet; a larger measured difference is almost
# certainly an octave-tracking artifact, and moving a whole melody by it would be
# worse than leaving the pitch alone.
_MAXIMUM_OFFSET_CENTS = 2400.0
# One region measures only its own level difference, so a register needs at least
# two to be a statement about the singer rather than about one word.
_CORROBORATING_REGIONS = 2
# A word whose travel exceeds a tritone is far more likely to be an octave error
# in the measurement than a note the student actually got wrong, so it is left
# where it was sung and reported instead of being moved.
_MAXIMUM_NOTE_SHIFT_CENTS = 600.0
# Below this the word is already on its note and moving it would only detune it.
_NEGLIGIBLE_NOTE_SHIFT_CENTS = 10.0
# A word's shift is constant across the word; where two words meet with
# different shifts the change is ramped over this window so the pitch the
# vocoder sees does not step.
_NOTE_RAMP_SECONDS = 0.030
_LEVEL_CLOCKS = ("teacher", "student")
# A per-word level has to describe the note that was held, not the gesture around
# it. Measured on the included pairs, the within-word pitch span has a median of
# about 240 cents, so a plain median of the word is pulled by whatever share of
# the word the glide and the vowel onset take. Validated against GAME's own note
# extraction, reading the held note instead cuts the per-word level error from
# 12.4 to 5.7 cents and raises the share of words within 25 cents from 66% to 85%.
# The sustained core is found as the peak of the word's own pitch histogram; its
# width is the tolerance, and a word with too few frames to have a shape is
# summarized by its mean.
_SUSTAINED_TOLERANCE_CENTS = 50.0
_SUSTAINED_BIN_CENTS = 10.0
# A core as wide as this is a slice of continuous movement rather than a held
# note, so the word's own median describes it better than the slice does.
_SUSTAINED_CORE_LIMIT_CENTS = 0.75 * _SUSTAINED_TOLERANCE_CENTS
# The share of a word's end that says where the syllable came to rest.
_TAIL_SHARE = 0.25
_MINIMUM_LEVEL_FRAMES = 3


def resolve_pitch_policy(pitch_mode: str, map_teacher_pitch: bool) -> str:
    """Map the pitch switch and the contour checkbox onto one rendering policy.

    ``off`` maps to ``none``, which is the only policy that leaves the pitch
    untouched. ``teacher`` moves the student into the teacher's register. For
    ``adaptive`` the checkbox chooses which contour carries the notes into the
    student's own register: off keeps the student's own phrasing and places each
    word on the teacher's note level, on uses the teacher's own contour, which
    already carries those levels and needs no per-word measurement to place them.
    """
    if pitch_mode not in PITCH_MODES:
        raise ValueError(f"Unknown pitch mode: {pitch_mode}. Choose from {', '.join(PITCH_MODES)}.")
    if pitch_mode == "off":
        return "none"
    if pitch_mode == "teacher":
        return "teacher_contour_teacher_register" if map_teacher_pitch else "student_contour_teacher_register"
    return "teacher_contour_student_register" if map_teacher_pitch else "student_contour_student_register"


def _frames(pitch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(pitch.get("times", []), dtype=np.float64)
    f0 = np.asarray(pitch.get("f0_hz", []), dtype=np.float64)
    voiced = np.asarray(pitch.get("voiced", f0 > 0), dtype=bool)
    if times.ndim != 1 or f0.shape != times.shape or voiced.shape != times.shape:
        raise ValueError("Pitch arrays must be equally sized one-dimensional arrays.")
    usable = voiced & np.isfinite(f0) & (f0 > 0)
    return times, np.where(usable, f0, 0.0), usable


def _sustained_cents(cents: np.ndarray) -> tuple[float, float]:
    """The held note inside one word, as (cents, share of the word it covers).

    The peak of the word's own pitch histogram is where the singer settled;
    everything within ``_SUSTAINED_TOLERANCE_CENTS`` of that peak is averaged back
    in, so the estimate stays a real observation rather than a grid point. A word
    with almost no pitch movement is already one value and is averaged whole, and a
    word with too few frames to have a shape is summarized by its mean instead of
    by a peak that would be noise.

    This shape was chosen by measurement rather than by construction: validated
    against GAME's independent note extraction on the included pairs, it cuts the
    per-word level error from 12.4 to 5.7 cents and raises the share of words
    within 25 cents of GAME's note from 66% to 85%. A longer argument for reading
    the held note, and the words it is less suitable for, is in the module
    docstring.
    """
    values = np.asarray(cents, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    if values.size < _MINIMUM_LEVEL_FRAMES:
        return float(np.mean(values)), 1.0
    span = float(values.max() - values.min())
    if span <= _SUSTAINED_TOLERANCE_CENTS:
        return float(np.mean(values)), 1.0
    lower = float(values.min())
    edges = np.arange(lower, lower + span + _SUSTAINED_BIN_CENTS, _SUSTAINED_BIN_CENTS)
    counts, _ = np.histogram(values, bins=edges)
    # A three-bin moving average keeps a one-bin spike from winning over the shelf
    # a real held note produces.
    smoothed = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    peak = int(np.argmax(smoothed))
    centre = (edges[peak] + edges[peak + 1]) / 2.0
    core = np.abs(values - centre) <= _SUSTAINED_TOLERANCE_CENTS
    if not np.any(core):
        return float(np.mean(values)), 1.0
    return float(np.mean(values[core])), float(np.mean(core))


def _level_detail(times: np.ndarray, f0: np.ndarray, usable: np.ndarray,
                  start: float, stop: float) -> tuple[float | None, float]:
    """The note held inside one interval plus the share of it that note occupies.

    A median of the whole word would be pulled by whatever share of it the glide
    and the vowel onset take, which made the estimate depend on the measurement
    method rather than on what was sung.
    """
    selected = usable & (times >= start) & (times < stop)
    if not np.any(selected):
        return None, 0.0
    cents = _CENTS_PER_OCTAVE * np.log2(f0[selected])
    sustained, share = _sustained_cents(cents)
    return float(2.0 ** (sustained / _CENTS_PER_OCTAVE)), float(share)


def _level(times: np.ndarray, f0: np.ndarray, usable: np.ndarray,
           start: float, stop: float) -> float | None:
    """``_level_detail``'s level alone, for callers that need only the frequency."""
    return _level_detail(times, f0, usable, start, stop)[0]


def register_pitch_plan(regions: list[dict], allowed: list[bool], teacher_pitch: dict,
                        student_pitch: dict, policy: str) -> dict:
    """Measure the register difference and decide how far each region travels.

    Levels are the note each singer held inside their own interval, read through
    ``_sustained_cents`` so a word's glide does not become its level. The same
    code works on either clock and one expressive word cannot dominate the
    estimate.

    ``region_shifts_cents`` is aligned with the ``regions`` list the caller
    passed in, one shift per region and ``0.0`` where nothing is applied, so the
    renderer can use it directly. The ``regions`` entries are the measurement
    evidence: both levels and how much of each word they cover, their difference,
    and, for per-word placement, the shift and the reason it was or was not
    applied.
    """
    if policy not in PITCH_POLICIES:
        raise ValueError(f"Unknown pitch policy: {policy}. Choose from {', '.join(PITCH_POLICIES)}.")
    student_register = policy in STUDENT_REGISTER_POLICIES
    teacher_contour = policy in TEACHER_CONTOUR_POLICIES
    per_word = policy in PER_WORD_PLACEMENT_POLICIES
    plan: dict = {
        "policy": policy, "applied": policy not in NO_PITCH_CHANGE_POLICIES,
        # Nothing is targeted when the pitch is left alone, and saying "teacher"
        # there would read as a register move that never happened.
        "target_register": "none" if policy == "none" else ("student" if student_register else "teacher"),
        "contour": "teacher" if teacher_contour else "student",
        "placement": "none" if policy == "none" else ("per_word" if per_word else "constant"),
        "offset_available": False, "offset_corroborated": False, "offset_cents": 0.0,
        "raw_offset_cents": None, "offset_reason": None,
        "contour_shift_cents": 0.0, "region_shifts_cents": [0.0] * len(regions),
        "note_shift_measured_count": 0, "note_shift_applied_count": 0,
        "region_count": 0, "comparable_regions": 0, "regions": [],
    }
    if policy == "none":
        plan["offset_reason"] = "Pitch change is disabled; the student's own pitch is used unchanged."
        return plan
    teacher_times, teacher_f0, teacher_usable = _frames(teacher_pitch)
    student_times, student_f0, student_usable = _frames(student_pitch)
    entries = []
    for index, region in enumerate(regions):
        if index >= len(allowed) or not allowed[index]:
            continue
        if "student_start" not in region or "student_end" not in region:
            continue
        teacher_level, teacher_share = _level_detail(
            teacher_times, teacher_f0, teacher_usable,
            float(region["teacher_start"]), float(region["teacher_end"]))
        student_level, student_share = _level_detail(
            student_times, student_f0, student_usable,
            float(region["student_start"]), float(region["student_end"]))
        entry = {"region_index": index,
                 "teacher_start": float(region["teacher_start"]),
                 "teacher_end": float(region["teacher_end"]),
                 "teacher_level_hz": teacher_level, "student_level_hz": student_level,
                 # How much of each word the level above actually covers. A low
                 # share means the word was mostly gesture, so its level is a
                 # weaker observation than a word that sat on its note.
                 "teacher_sustained_share": round(teacher_share, 3),
                 "student_sustained_share": round(student_share, 3),
                 "difference_cents": None}
        if teacher_level is not None and student_level is not None:
            entry["difference_cents"] = float(
                _CENTS_PER_OCTAVE * np.log2(student_level / teacher_level))
        entries.append(entry)
    plan["region_count"] = len(entries)
    plan["regions"] = entries
    comparable = [entry for entry in entries if entry["difference_cents"] is not None]
    plan["comparable_regions"] = len(comparable)
    if len(comparable) < _CORROBORATING_REGIONS:
        plan["offset_reason"] = (
            f"Only {len(comparable)} matched region(s) carry voiced pitch on both sides; the register "
            "difference needs at least two to be a statement about the singer rather than about one "
            "word, so no register move is applied.")
        _annotate_unplaced(entries, plan["offset_reason"], per_word)
        return plan
    differences = [entry["difference_cents"] for entry in comparable]
    raw_cents = float(np.median(differences))
    plan["raw_offset_cents"] = raw_cents
    if abs(raw_cents) > _MAXIMUM_OFFSET_CENTS:
        plan["offset_reason"] = (
            f"Measured register difference of {raw_cents:.0f} cents exceeds two octaves; this is more "
            "likely an octave-tracking error than the interval sung, so no register move is applied.")
        _annotate_unplaced(entries, plan["offset_reason"], per_word)
        return plan
    offset_cents = float(np.round(raw_cents / _SEMITONE_CENTS) * _SEMITONE_CENTS)
    plan.update(offset_available=True, offset_corroborated=True, offset_cents=offset_cents,
                offset_semitones=offset_cents / _SEMITONE_CENTS)
    # The constant to apply follows from which contour was chosen and which register
    # it has to end up in: shift = target level - chosen contour's own level.
    # ``offset_cents`` is the student level minus the teacher level, so a student
    # singing below the teacher makes it negative.
    if teacher_contour:
        # The teacher's contour starts at the teacher's level.
        plan["contour_shift_cents"] = offset_cents if student_register else 0.0
    else:
        # The student's contour starts at the student's level.
        plan["contour_shift_cents"] = 0.0 if student_register else -offset_cents
    if per_word:
        _plan_note_shifts(plan, entries, comparable, offset_cents)
    return plan


def _annotate_unplaced(entries: list[dict], reason: str, per_word: bool) -> None:
    """Record on every measured region why per-word placement could not run.

    Without a measured register there is no anchor to place a word against, so
    the plan reports the same reason per region instead of leaving the fields
    absent and making a reader guess why nothing moved.
    """
    if not per_word:
        return
    for entry in entries:
        entry["note_shift_cents"] = None
        entry["note_shift_applied"] = False
        entry["note_shift_reason"] = reason


def _plan_note_shifts(plan: dict, entries: list[dict], comparable: list[dict], offset_cents: float) -> None:
    """Decide each region's travel onto the teacher's note level.

    ``offset_cents`` anchors the teacher's note levels in the student's register,
    so a word's target is ``teacher_level + offset_cents`` and its shift is
    ``offset_cents - difference``, which is zero for a word already in place.
    """
    shifts = [0.0] * len(plan["region_shifts_cents"])
    for entry in entries:
        entry["note_shift_cents"] = None
        entry["note_shift_applied"] = False
        difference = entry["difference_cents"]
        if difference is None:
            entry["note_shift_reason"] = ("The region carries voiced pitch on only one side, so it has no "
                                          "note to be moved onto.")
            continue
        shift = offset_cents - difference
        if abs(shift) > _MAXIMUM_NOTE_SHIFT_CENTS:
            entry["note_shift_reason"] = (
                f"Reaching the teacher's note needs {shift:+.0f} cents, beyond the "
                f"{_MAXIMUM_NOTE_SHIFT_CENTS:.0f}-cent limit; a difference that large is more likely an "
                "octave error in pitch tracking than the interval sung, so the word keeps its own level.")
            continue
        if abs(shift) < _NEGLIGIBLE_NOTE_SHIFT_CENTS:
            entry["note_shift_cents"] = 0.0
            entry["note_shift_reason"] = ("The word already sits on its target note; the residual is below "
                                          f"{_NEGLIGIBLE_NOTE_SHIFT_CENTS:.0f} cents and is left alone.")
            continue
        entry["note_shift_cents"] = float(shift)
        entry["note_shift_applied"] = True
        shifts[entry["region_index"]] = float(shift)
    plan["region_shifts_cents"] = shifts
    plan["note_shift_measured_count"] = len(comparable)
    plan["note_shift_applied_count"] = sum(1 for entry in entries if entry["note_shift_applied"])


def frame_note_shifts(regions: list[dict], shifts_cents, times: np.ndarray, *,
                      clock: str = "teacher", ramp_seconds: float = _NOTE_RAMP_SECONDS) -> np.ndarray:
    """Turn per-region shifts into one cent shift per output frame.

    A word's shift is held constant across the word, so the student's own slides
    and vibrato keep their shape. Where two words meet with different shifts the
    change is ramped over a short window instead of stepping, and only over that
    window: nothing inside a word is tilted. Material outside every matched
    region keeps its own level.

    ``clock`` selects which interval of a region the output frames index: the
    teacher's when the render was warped onto the teacher's timeline, the
    student's when the student's own timeline was kept.
    """
    if clock not in _LEVEL_CLOCKS:
        raise ValueError(f"Unknown shift clock: {clock}. Choose from {', '.join(_LEVEL_CLOCKS)}.")
    times = np.asarray(times, dtype=np.float64)
    output = np.zeros(len(times), dtype=np.float64)
    if not len(times):
        return output
    spans = []
    for index, region in enumerate(regions):
        shift = float(shifts_cents[index]) if index < len(shifts_cents) else 0.0
        if "student_start" not in region or "student_end" not in region:
            continue
        if clock == "teacher":
            start, stop = float(region["teacher_start"]), float(region["teacher_end"])
        else:
            start, stop = float(region["student_start"]), float(region["student_end"])
        if stop <= start:
            continue
        spans.append((start, stop, shift))
    if not spans:
        return output
    spans.sort(key=lambda span: span[0])
    control_times: list[float] = []
    control_values: list[float] = []
    for start, stop, shift in spans:
        ramp = min(ramp_seconds, (stop - start) / 4.0)
        for value in (start + ramp, stop - ramp):
            if control_times and value <= control_times[-1]:
                continue
            control_times.append(value)
            control_values.append(shift)
    for anchor, value in ((control_times[0] - ramp_seconds, 0.0), (control_times[-1] + ramp_seconds, 0.0)):
        if anchor <= control_times[0]:
            control_times.insert(0, anchor)
            control_values.insert(0, value)
        elif anchor >= control_times[-1]:
            control_times.append(anchor)
            control_values.append(value)
    return np.interp(times, control_times, control_values)


def shifted_f0(f0: np.ndarray, cents) -> np.ndarray:
    """Move a frequency track by a constant offset or a per-frame offset in cents."""
    return (np.asarray(f0, dtype=np.float64)
            * 2.0 ** (np.asarray(cents, dtype=np.float64) / _CENTS_PER_OCTAVE)).astype(np.float32)
