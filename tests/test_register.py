from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from inference.correction.register import (_sustained_cents,
                                           NO_PITCH_CHANGE_POLICIES, PER_WORD_PLACEMENT_POLICIES,
                                           PITCH_MODES, PITCH_POLICIES, STUDENT_REGISTER_POLICIES,
                                           TEACHER_CONTOUR_POLICIES, frame_note_shifts,
                                           register_pitch_plan, resolve_pitch_policy, shifted_f0)
from inference.correction.renderer import render_student
from test_vocoder import RecordingVocoder

LOW = 440.0 * 2 ** (-7 / 12)
TEACHER_LEVELS = [440.0] * 3
IN_TUNE = [LOW] * 3
# The middle word is a minor third sharp of where it should be, in the student's
# own register, so adaptive placement has a well-defined note to correct.
SHARP_MIDDLE = [LOW, LOW * 2 ** (3 / 12), LOW]


def pitch_track(times, f0, voiced=None):
    f0 = np.asarray(f0, dtype=float)
    voiced = (f0 > 0) if voiced is None else np.asarray(voiced, dtype=bool)
    return {"times": np.asarray(times, dtype=float).tolist(), "f0_hz": f0.tolist(),
            "voiced": voiced.tolist()}


def level_edges(bounds, levels):
    times, f0 = [], []
    for (start, stop), level in zip(bounds, levels):
        for value in np.linspace(start + 0.02, stop - 0.02, 5):
            times.append(value)
            f0.append(0.0 if level is None else float(level))
    return pitch_track(times, f0)


def regions(count):
    return [{"teacher_start": float(index), "teacher_end": float(index + 1),
             "student_start": float(index), "student_end": float(index + 1)}
            for index in range(count)]


def analysis(count, f0, *, sample_rate=44100, hop=512):
    times = (np.arange((count + hop - 1) // hop) + 0.5) * hop / sample_rate
    return {"sample_rate": sample_rate, "sample_count": count,
            "pitch": pitch_track(times, np.full(len(times), f0))}


class PolicyResolutionTests(unittest.TestCase):
    def test_the_switch_selects_the_policy(self):
        self.assertEqual(resolve_pitch_policy("off", False), "none")
        self.assertEqual(resolve_pitch_policy("off", True), "none")
        self.assertEqual(resolve_pitch_policy("teacher", False), "student_contour_teacher_register")
        self.assertEqual(resolve_pitch_policy("teacher", True), "teacher_contour_teacher_register")
        self.assertEqual(resolve_pitch_policy("adaptive", False), "student_contour_student_register")
        self.assertEqual(resolve_pitch_policy("adaptive", True), "teacher_contour_student_register")

    def test_the_contour_checkbox_only_applies_where_there_is_a_contour_to_choose(self):
        self.assertEqual(resolve_pitch_policy("off", False), resolve_pitch_policy("off", True))
        for mode in ("teacher", "adaptive"):
            self.assertNotEqual(resolve_pitch_policy(mode, False), resolve_pitch_policy(mode, True), mode)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_pitch_policy("loud", False)
        self.assertEqual(PITCH_MODES, ("off", "teacher", "adaptive"))

    def test_switching_the_pitch_off_promises_no_pitch_change_at_all(self):
        # This is the promise the reported bug broke: only changing the timing must
        # not move the pitch, so the policy is declared pitch-preserving.
        self.assertIn(resolve_pitch_policy("off", False), NO_PITCH_CHANGE_POLICIES)
        self.assertIn(resolve_pitch_policy("off", True), NO_PITCH_CHANGE_POLICIES)
        plan = register_pitch_plan(regions(2), [True, True],
                                   level_edges([(0, 1), (1, 2)], [440.0, 440.0]),
                                   level_edges([(0, 1), (1, 2)], [293.33, 293.33]), "none")
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["contour_shift_cents"], 0.0)
        self.assertEqual(plan["region_shifts_cents"], [0.0, 0.0])

    def test_each_policy_declares_its_register_and_contour(self):
        cases = (("off", False), ("off", True), ("teacher", False), ("teacher", True),
                 ("adaptive", False), ("adaptive", True))
        for mode, map_teacher in cases:
            policy = resolve_pitch_policy(mode, map_teacher)
            plan = register_pitch_plan(regions(2), [True, True],
                                       level_edges([(0, 1), (1, 2)], [440.0, 440.0]),
                                       level_edges([(0, 1), (1, 2)], [LOW, LOW]), policy)
            target = "none" if policy == "none" else ("teacher" if mode == "teacher" else "student")
            self.assertEqual(plan["target_register"], target, policy)
            wants_teacher_contour = map_teacher and mode != "off"
            self.assertEqual(plan["contour"], "teacher" if wants_teacher_contour else "student", policy)
            if policy == "none":
                # The disabled policy targets nothing and is deliberately in
                # neither register set, so a reader cannot mistake it for a move.
                self.assertNotIn(policy, STUDENT_REGISTER_POLICIES)
                continue
            self.assertEqual(policy in STUDENT_REGISTER_POLICIES, mode != "teacher", policy)
            self.assertEqual(policy in TEACHER_CONTOUR_POLICIES, wants_teacher_contour, policy)
        self.assertEqual(set(PITCH_POLICIES), {"none", "student_contour_student_register",
                                              "teacher_contour_student_register",
                                              "student_contour_teacher_register",
                                              "teacher_contour_teacher_register"})


class SustainedNoteTests(unittest.TestCase):
    """The per-word level reads the note that was held, not the word's median.

    The within-word span median of about 240 cents is why the reading matters."""

    def test_a_plain_held_note_is_returned_at_its_own_level(self):
        cents = np.full(30, 500.0)
        level, share = _sustained_cents(cents)
        self.assertAlmostEqual(level, 500.0, places=6)
        self.assertEqual(share, 1.0)

    def test_a_small_deviation_is_still_one_note(self):
        cents = 700.0 + np.linspace(-20, 20, 21)
        level, _ = _sustained_cents(cents)
        self.assertAlmostEqual(level, 700.0, delta=1.0)

    def test_a_vowel_that_occupies_most_of_the_word_is_the_level(self):
        # The measured case: the word is mostly the note, with a glide and an onset
        # taking the rest. A plain median fails here because the glide pulls it down.
        word = np.r_[np.linspace(-250.0, 0.0, 12), np.full(30, 0.0)]
        level, share = _sustained_cents(word)
        self.assertAlmostEqual(level, 0.0, delta=10.0)
        self.assertGreater(share, 0.5)

    def test_the_level_does_not_depend_on_which_half_is_the_glide(self):
        # A rising glide and a falling one onto the same note must read the same
        # note; a median would read one of them flat and the other sharp.
        rising = np.r_[np.linspace(-250.0, 0.0, 12), np.full(30, 0.0)]
        falling = np.r_[np.linspace(250.0, 0.0, 12), np.full(30, 0.0)]
        self.assertAlmostEqual(_sustained_cents(rising)[0], 0.0, delta=10.0)
        self.assertAlmostEqual(_sustained_cents(falling)[0], 0.0, delta=10.0)

    def test_a_word_spent_mostly_on_the_glide_is_the_known_limitation(self):
        # An extreme shape: the glide is longer than the note it lands on, so the
        # histogram peak sits inside the glide. Recorded rather than papered over,
        # because it is why per-word results carry their own measured error.
        word = np.r_[np.linspace(-200.0, 0.0, 17), np.full(13, 0.0)]
        level, _ = _sustained_cents(word)
        self.assertLess(level, 0.0)
        self.assertGreater(level, -200.0)
        self.assertGreater(abs(level), 5.0)

    def test_a_word_with_too_few_frames_does_not_have_a_peak_to_trust(self):
        level, share = _sustained_cents(np.array([100.0, 500.0]))
        self.assertAlmostEqual(level, 300.0, places=6)
        self.assertEqual(share, 1.0)

    def test_an_empty_word_has_no_level(self):
        self.assertEqual(_sustained_cents(np.array([])), (0.0, 0.0))

    def test_nonfinite_frames_are_ignored(self):
        level, _ = _sustained_cents(np.array([440.0, np.nan, 440.0, np.inf]))
        self.assertAlmostEqual(level, 440.0, places=6)


class RegisterPlanTests(unittest.TestCase):
    def test_a_consistent_register_difference_becomes_one_constant_move(self):
        teacher = level_edges([(0, 1), (1, 2)], [440.0, 440.0])
        student = level_edges([(0, 1), (1, 2)], [LOW, LOW])
        plan = register_pitch_plan(regions(2), [True, True], teacher, student,
                                   "student_contour_teacher_register")
        self.assertTrue(plan["offset_available"])
        self.assertTrue(plan["offset_corroborated"])
        self.assertAlmostEqual(plan["offset_cents"], -700.0, delta=1.0)
        self.assertAlmostEqual(plan["contour_shift_cents"], 700.0, delta=1.0)
        self.assertEqual(plan["placement"], "constant")

    def test_the_absolute_policy_applies_no_move(self):
        # The teacher's contour in the teacher's register is already where it
        # belongs, so the constant is zero -- the bug my own test caught.
        teacher = level_edges([(0, 1), (1, 2)], [440.0, 440.0])
        student = level_edges([(0, 1), (1, 2)], [LOW, LOW])
        plan = register_pitch_plan(regions(2), [True, True], teacher, student,
                                   "teacher_contour_teacher_register")
        self.assertAlmostEqual(plan["offset_cents"], -700.0, delta=1.0)
        self.assertEqual(plan["contour_shift_cents"], 0.0)

    def test_offsets_are_quantized_to_whole_semitones(self):
        teacher = level_edges([(0, 1), (1, 2)], [440.0, 440.0])
        student = level_edges([(0, 1), (1, 2)], [440.0 * 2 ** (-1.37 / 12)] * 2)
        plan = register_pitch_plan(regions(2), [True, True], teacher, student,
                                   "student_contour_teacher_register")
        self.assertAlmostEqual(plan["offset_cents"], -100.0, delta=1e-6)
        self.assertAlmostEqual(plan["raw_offset_cents"], -137.0, delta=1.0)

    def test_a_wild_octave_error_does_not_move_the_whole_melody(self):
        teacher = level_edges([(0, 1), (1, 2)], [440.0, 440.0])
        student = level_edges([(0, 1), (1, 2)], [55.0, 55.0])
        plan = register_pitch_plan(regions(2), [True, True], teacher, student,
                                   "student_contour_teacher_register")
        self.assertFalse(plan["offset_available"])
        self.assertEqual(plan["contour_shift_cents"], 0.0)
        self.assertIn("two octaves", plan["offset_reason"])

    def test_a_single_region_cannot_establish_a_register(self):
        plan = register_pitch_plan(regions(1), [True], level_edges([(0, 1)], [440.0]),
                                   level_edges([(0, 1)], [LOW]),
                                   "student_contour_teacher_register")
        self.assertFalse(plan["offset_available"])
        self.assertIn("at least two", plan["offset_reason"])
        self.assertEqual(plan["contour_shift_cents"], 0.0)

    def test_a_single_region_cannot_place_notes_either(self):
        plan = register_pitch_plan(regions(1), [True], level_edges([(0, 1)], [440.0]),
                                   level_edges([(0, 1)], [LOW]), "student_contour_student_register")
        self.assertFalse(plan["offset_available"])
        self.assertEqual(plan["region_shifts_cents"], [0.0])
        self.assertEqual(plan["note_shift_applied_count"], 0)

    def test_regions_without_voiced_pitch_cannot_measure_a_register(self):
        teacher = level_edges([(0, 1), (1, 2)], [None, None])
        student = level_edges([(0, 1), (1, 2)], [LOW, LOW])
        plan = register_pitch_plan(regions(2), [True, True], teacher, student,
                                   "student_contour_teacher_register")
        self.assertFalse(plan["offset_available"])
        self.assertEqual(plan["comparable_regions"], 0)
        self.assertEqual(plan["region_count"], 2)

    def test_only_matched_regions_are_measured(self):
        plan = register_pitch_plan(regions(2), [True, False],
                                   level_edges([(0, 1), (1, 2)], [440.0, 440.0]),
                                   level_edges([(0, 1), (1, 2)], [LOW, LOW]),
                                   "student_contour_teacher_register")
        self.assertEqual(plan["region_count"], 1)

    def test_the_disabled_policy_reports_a_reason_and_no_move(self):
        plan = register_pitch_plan(regions(1), [True], level_edges([(0, 1)], [440.0]),
                                   level_edges([(0, 1)], [220.0]), "none")
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["contour_shift_cents"], 0.0)
        self.assertIn("disabled", plan["offset_reason"])


class AdaptivePlacementTests(unittest.TestCase):
    def test_only_the_adaptive_policy_places_notes_individually(self):
        self.assertEqual(PER_WORD_PLACEMENT_POLICIES, ("student_contour_student_register",))

    def test_an_in_tune_student_is_not_moved_at_all(self):
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], IN_TUNE),
                                   "student_contour_student_register")
        self.assertTrue(plan["offset_available"])
        self.assertAlmostEqual(plan["offset_cents"], -700.0, delta=1.0)
        self.assertEqual(plan["region_shifts_cents"], [0.0, 0.0, 0.0])
        self.assertEqual(plan["note_shift_applied_count"], 0)
        self.assertEqual(plan["placement"], "per_word")

    def test_a_sharp_word_is_moved_down_by_exactly_its_own_error(self):
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], SHARP_MIDDLE),
                                   "student_contour_student_register")
        # The offset is the median of the differences, so one wrong word does not
        # move the register estimate.
        self.assertAlmostEqual(plan["offset_cents"], -700.0, delta=1.0)
        self.assertAlmostEqual(plan["region_shifts_cents"][1], -300.0, delta=1.0)
        self.assertEqual(plan["region_shifts_cents"][0], 0.0)
        self.assertEqual(plan["region_shifts_cents"][2], 0.0)
        self.assertEqual(plan["note_shift_applied_count"], 1)

    def test_the_placed_level_is_inside_the_students_own_range(self):
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], SHARP_MIDDLE),
                                   "student_contour_student_register")
        entry = plan["regions"][1]
        level = entry["student_level_hz"] * 2 ** (plan["region_shifts_cents"][1] / 1200.0)
        self.assertAlmostEqual(level, LOW, delta=1.0)
        self.assertLess(level, 440.0)

    def test_a_move_beyond_the_safe_limit_is_refused_and_reported(self):
        # An octave-tracking slip looks like a 1200-cent error; moving the word
        # that far would be worse than leaving it, so it is left and explained.
        student_levels = [LOW, LOW * 2, LOW]
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], student_levels),
                                   "student_contour_student_register")
        self.assertEqual(plan["region_shifts_cents"][1], 0.0)
        self.assertEqual(plan["note_shift_applied_count"], 0)
        self.assertIn("octave error", plan["regions"][1]["note_shift_reason"])

    def test_a_negligible_residual_is_not_applied(self):
        student_levels = [LOW, LOW * 2 ** (0.03 / 12), LOW]
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], student_levels),
                                   "student_contour_student_register")
        self.assertEqual(plan["region_shifts_cents"][1], 0.0)
        self.assertIn("already sits", plan["regions"][1]["note_shift_reason"])

    def test_a_region_voiced_on_one_side_has_no_note_to_reach(self):
        # Enough comparable regions for the register to be measurable, so the
        # refusal is about this region's own missing level, not about the register.
        plan = register_pitch_plan(regions(3), [True] * 3,
                                   level_edges([(0, 1), (1, 2), (2, 3)], [440.0] * 3),
                                   level_edges([(0, 1), (1, 2), (2, 3)], [None, LOW, LOW]),
                                   "student_contour_student_register")
        self.assertTrue(plan["offset_available"])
        self.assertEqual(plan["region_shifts_cents"], [0.0, 0.0, 0.0])
        self.assertIn("only one side", plan["regions"][0]["note_shift_reason"])

    def test_region_shifts_are_aligned_with_the_caller_regions(self):
        # The renderer indexes the shift list with its own region list, so a
        # rejected region must keep its slot rather than shifting its neighbours.
        plan = register_pitch_plan(regions(3), [True, False, True],
                                   level_edges([(0, 1), (1, 2), (2, 3)], TEACHER_LEVELS),
                                   level_edges([(0, 1), (1, 2), (2, 3)], SHARP_MIDDLE),
                                   "student_contour_student_register")
        self.assertEqual(len(plan["region_shifts_cents"]), 3)
        self.assertEqual(plan["region_shifts_cents"][1], 0.0)
        self.assertEqual([entry["region_index"] for entry in plan["regions"]], [0, 2])


class FrameShiftTests(unittest.TestCase):
    def test_a_word_holds_one_shift_across_its_whole_length(self):
        times = (np.arange(400) + 0.5) * 0.01
        shifts = [200.0]
        track = frame_note_shifts(regions(1), shifts, times)
        inside = (times > 0.05) & (times < 0.95)
        np.testing.assert_allclose(track[inside], 200.0)

    def test_material_outside_every_region_keeps_its_own_level(self):
        times = (np.arange(400) + 0.5) * 0.01
        track = frame_note_shifts(regions(1), [200.0], times)
        outside = (times < -0.01) | (times > 1.01)
        np.testing.assert_allclose(track[outside], 0.0)

    def test_different_shifts_are_ramped_instead_of_stepping(self):
        # A step in F0 across a word joint clicks; the ramp spreads it over a short
        # window but leaves each word's interior flat. Two one-second words.
        times = (np.arange(2400) + 0.5) * 0.001
        track = frame_note_shifts(regions(2), [100.0, -100.0], times)

        def at(second):
            return float(track[int(np.argmin(np.abs(times - second)))])

        self.assertAlmostEqual(at(1.0), 0.0, delta=20.0)    # the joint itself
        self.assertAlmostEqual(at(0.5), 100.0, delta=1e-6)  # first word, flat
        self.assertAlmostEqual(at(1.5), -100.0, delta=1e-6) # second word, flat
        self.assertAlmostEqual(at(0.02), 80.0, delta=20.0)  # ramped in
        self.assertAlmostEqual(at(2.1), 0.0, delta=1e-6)    # back to no shift

    def test_the_student_clock_reads_the_students_intervals(self):
        # Without a timing change the frames index the student's own timeline, so
        # the shift has to be read from the student's interval or it lands on the
        # wrong word.
        layout = [{"teacher_start": 0.0, "teacher_end": 0.5, "student_start": 2.0, "student_end": 3.0}]
        times = np.array([0.25, 2.5])
        np.testing.assert_allclose(frame_note_shifts(layout, [300.0], times, clock="teacher"), [300.0, 0.0])
        np.testing.assert_allclose(frame_note_shifts(layout, [300.0], times, clock="student"), [0.0, 300.0])

    def test_an_unknown_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            frame_note_shifts(regions(1), [0.0], np.arange(3) * 0.01, clock="wall")

    def test_shifted_f0_accepts_a_per_frame_track(self):
        track = shifted_f0(np.array([100.0, 100.0]), np.array([0.0, 1200.0]))
        np.testing.assert_allclose(track, [100.0, 200.0], rtol=1e-6)

    def test_shifted_f0_still_accepts_a_constant(self):
        np.testing.assert_allclose(shifted_f0(np.array([100.0, 200.0]), 1200.0), [200.0, 400.0], rtol=1e-6)


class RendererOptionTests(unittest.TestCase):
    def fixture(self, student_levels, teacher_levels=None):
        teacher = analysis(132300, 440.0)
        if teacher_levels is not None:
            teacher["pitch"] = level_edges([(index, index + 1) for index in range(3)], teacher_levels)
        student = analysis(264600, student_levels[0])
        student["pitch"] = level_edges([(index * 2, index * 2 + 2) for index in range(3)],
                                       list(student_levels))
        plan = {"regions": [{"teacher_start": float(index), "teacher_end": float(index + 1),
                             "student_start": float(index * 2), "student_end": float(index * 2 + 2),
                             "mode": "synthesize"} for index in range(3)]}
        return teacher, student, plan

    def render(self, student_levels=None, teacher_levels=None, **options):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture(student_levels or [LOW, LOW, LOW], teacher_levels)
        wave, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan,
                                          vocoder, **options)
        return vocoder, wave, diagnostics

    def test_the_default_leaves_the_pitch_exactly_as_sung(self):
        # The reported bug: this case moved 14 of 17 words. It must move nothing.
        vocoder, _, diagnostics = self.render()
        self.assertEqual(diagnostics["pitch_policy"], "none")
        self.assertFalse(diagnostics["register_pitch"]["applied"])
        self.assertEqual(diagnostics["register_pitch"]["contour_shift_cents"], 0.0)
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertAlmostEqual(float(np.median(voiced)), LOW, places=3)

    def test_a_wrong_note_is_not_corrected_when_the_switch_is_off(self):
        # A student who simply sings one word at another pitch keeps that choice
        # unless the user asks for a correction; making it silently is what made a
        # timing-only job change the pitch.
        vocoder, _, diagnostics = self.render([LOW, LOW / 1.5, LOW])
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertGreater(float(np.max(voiced)) / float(np.min(voiced)), 1.2)
        self.assertEqual(diagnostics["register_pitch"]["contour_shift_cents"], 0.0)

    def test_timing_can_change_without_touching_pitch(self):
        vocoder, wave, diagnostics = self.render()
        self.assertEqual(len(wave), 132300)
        self.assertTrue(diagnostics["timing_changed"])
        # The synthesised pitch is the student's own, frame for frame. The renderer
        # is asked to re-render the same audio, so the expected track is whatever a
        # pitch-preserving reference render produces from the same input.
        reference = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, reference,
                       mode="time_only")
        voiced = vocoder.last_f0 > 0
        self.assertTrue(np.any(voiced))
        np.testing.assert_allclose(vocoder.last_f0[voiced], reference.last_f0[voiced], rtol=1e-6)

    def test_pitch_can_change_without_touching_timing(self):
        vocoder, wave, diagnostics = self.render(change_timing=False)
        self.assertEqual(len(wave), 264600)
        self.assertFalse(diagnostics["timing_changed"])

    def test_the_teacher_mode_moves_the_student_into_the_teachers_register(self):
        vocoder, _, diagnostics = self.render(pitch_mode="teacher")
        self.assertEqual(diagnostics["pitch_policy"], "student_contour_teacher_register")
        self.assertAlmostEqual(diagnostics["register_pitch"]["contour_shift_cents"], 700.0, delta=1.0)
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertAlmostEqual(float(np.median(voiced)), 440.0, delta=1.0)

    def test_the_teacher_mode_with_the_contour_uses_the_teacher_curve(self):
        vocoder, _, diagnostics = self.render(pitch_mode="teacher", map_teacher_pitch=True)
        self.assertEqual(diagnostics["pitch_policy"], "teacher_contour_teacher_register")
        self.assertEqual(diagnostics["register_pitch"]["contour_shift_cents"], 0.0)
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertAlmostEqual(float(np.median(voiced)), 440.0, delta=1.0)

    def test_adaptive_placement_fixes_the_wrong_note_and_keeps_the_register(self):
        vocoder, _, diagnostics = self.render(SHARP_MIDDLE, TEACHER_LEVELS, pitch_mode="adaptive")
        self.assertEqual(diagnostics["pitch_policy"], "student_contour_student_register")
        self.assertEqual(diagnostics["register_pitch"]["note_shift_applied_count"], 1)
        track = vocoder.last_f0
        voiced = track > 0
        self.assertTrue(np.any(voiced))
        levels = np.percentile(track[voiced], [5, 95])
        self.assertLess(float(levels[1]), 440.0)
        self.assertLess(float(levels[1] / levels[0]), 1.12)
        uncorrected = RecordingVocoder()
        teacher, student, plan = self.fixture(SHARP_MIDDLE, TEACHER_LEVELS)
        render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, uncorrected,
                       mode="time_only")
        spread = uncorrected.last_f0[uncorrected.last_f0 > 0]
        self.assertGreater(float(np.percentile(spread, 95) / np.percentile(spread, 5)),
                           float(levels[1] / levels[0]) + 0.1)

    def test_adaptive_placement_leaves_an_in_tune_student_alone(self):
        vocoder, _, diagnostics = self.render(IN_TUNE, TEACHER_LEVELS, pitch_mode="adaptive")
        self.assertEqual(diagnostics["register_pitch"]["note_shift_applied_count"], 0)
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertAlmostEqual(float(np.median(voiced)), LOW, delta=1.0)

    def warbling_fixture(self):
        teacher = analysis(132300, 440.0)
        times = np.arange(300) * 0.01
        teacher["pitch"] = pitch_track(times, 440.0 * 2 ** (np.sin(times * 30) * 0.5 / 12))
        student = analysis(264600, LOW)
        student["pitch"] = pitch_track(times, np.full(len(times), LOW))
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.5, "student_start": 0.0,
                             "student_end": 1.5, "mode": "synthesize"},
                            {"teacher_start": 1.5, "teacher_end": 3.0, "student_start": 1.5,
                             "student_end": 3.0, "mode": "synthesize"}]}
        return teacher, student, plan

    def test_adaptive_placement_keeps_the_students_own_contour_by_default(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.warbling_fixture()
        render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                       pitch_mode="adaptive")
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        # A steady student stays steady: the teacher's half-semitone (about 900
        # cents peak-to-peak) warble would show up as a spread of hundreds of cents.
        spread = float(np.percentile(voiced, 95) / np.percentile(voiced, 5))
        self.assertLess(spread, 1.05, spread)

    def test_adaptive_with_the_contour_takes_the_teachers_own_movement(self):
        # The checkbox is meaningful in adaptive mode too: the teacher's contour
        # already carries the note levels, so it is transposed into the student's
        # register as one constant instead of placing words individually.
        vocoder = RecordingVocoder()
        teacher, student, plan = self.warbling_fixture()
        _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan,
                                        vocoder, pitch_mode="adaptive", map_teacher_pitch=True)
        self.assertEqual(diagnostics["pitch_policy"], "teacher_contour_student_register")
        self.assertEqual(diagnostics["register_pitch"]["placement"], "constant")
        self.assertEqual(diagnostics["register_pitch"]["note_shift_applied_count"], 0)
        voiced = vocoder.last_f0[vocoder.last_f0 > 0]
        self.assertGreater(float(np.percentile(voiced, 95) / np.percentile(voiced, 5)), 1.02)
        self.assertLess(float(np.percentile(voiced, 95)), 440.0)

    def test_the_contour_checkbox_changes_the_adaptive_render(self):
        without, _, first = self.render(SHARP_MIDDLE, TEACHER_LEVELS, pitch_mode="adaptive")
        with_contour, _, second = self.render(SHARP_MIDDLE, TEACHER_LEVELS, pitch_mode="adaptive",
                                              map_teacher_pitch=True)
        self.assertEqual(first["pitch_policy"], "student_contour_student_register")
        self.assertEqual(second["pitch_policy"], "teacher_contour_student_register")
        self.assertFalse(np.allclose(without.last_f0, with_contour.last_f0))
        for vocoder in (without, with_contour):
            voiced = vocoder.last_f0[vocoder.last_f0 > 0]
            self.assertLess(float(np.percentile(voiced, 95)), 440.0)

    def test_a_single_region_job_reports_that_no_register_was_established(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        plan["regions"] = plan["regions"][:1]
        _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                                        pitch_mode="teacher")
        self.assertFalse(diagnostics["register_pitch"]["offset_available"])
        self.assertTrue(any("pitch move was not applied" in warning
                            for warning in diagnostics["warnings"]))

    def test_the_absolute_policy_needs_no_register_and_says_nothing(self):
        # The teacher's own contour in the teacher's own register is already where it
        # belongs, so a refused measurement is not a failure here and must not be
        # reported as one.
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        plan["regions"] = plan["regions"][:1]
        _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                                        pitch_mode="teacher", map_teacher_pitch=True)
        self.assertFalse(diagnostics["register_pitch"]["offset_available"])
        self.assertFalse(any("was not applied" in warning for warning in diagnostics["warnings"]))

    def test_adaptive_mode_says_so_when_it_could_not_measure_a_register(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        plan["regions"] = plan["regions"][:1]
        _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                                        pitch_mode="adaptive")
        self.assertTrue(any("Adaptive pitch placement could not measure" in warning
                            for warning in diagnostics["warnings"]))

    def test_the_warning_count_says_how_many_words_were_moved(self):
        _, _, diagnostics = self.render(SHARP_MIDDLE, TEACHER_LEVELS, pitch_mode="adaptive")
        self.assertTrue(any("moved 1 of 3" in warning for warning in diagnostics["warnings"]))

    def test_exported_target_reports_the_options_and_the_move(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                           target_pitch_path=path, pitch_mode="teacher", map_teacher_pitch=True)
            with np.load(path) as saved:
                self.assertEqual(str(saved["pitch_policy"]), "teacher_contour_teacher_register")
                self.assertAlmostEqual(float(saved["register_offset_cents"]), -700.0, delta=1.0)
                self.assertTrue(bool(saved["register_offset_available"]))
                self.assertTrue(bool(saved["timing_changed"]))
                self.assertEqual(int(saved["note_shift_applied_count"]), 0)

    def test_an_adaptive_render_exports_its_per_word_count(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture(SHARP_MIDDLE, TEACHER_LEVELS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            render_student(np.ones(264600, dtype=np.float32), teacher, student, plan, vocoder,
                           target_pitch_path=path, pitch_mode="adaptive")
            with np.load(path) as saved:
                self.assertEqual(str(saved["pitch_policy"]), "student_contour_student_register")
                self.assertEqual(int(saved["note_shift_applied_count"]), 1)

    def test_experiments_keep_their_pitch_contracts(self):
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        for mode in ("identity", "time_only"):
            _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan,
                                            vocoder, mode=mode)
            self.assertEqual(diagnostics["pitch_policy"], "none", mode)

    def test_the_experiments_ignore_the_pitch_switch(self):
        # identity and time_only are the pitch-preserving halves of the A/B set, so
        # their behaviour is fixed by the mode name.
        vocoder = RecordingVocoder()
        teacher, student, plan = self.fixture([LOW, LOW, LOW])
        for mode in ("identity", "time_only"):
            _, diagnostics = render_student(np.ones(264600, dtype=np.float32), teacher, student, plan,
                                            vocoder, mode=mode, pitch_mode="teacher",
                                            map_teacher_pitch=True)
            self.assertEqual(diagnostics["pitch_policy"], "none", mode)


if __name__ == "__main__":
    unittest.main()
