"""Bilingual UI strings; widgets bind a refresh callable for live switching."""
from __future__ import annotations

current_language = "zh"

_STRINGS: dict[str, dict[str, str]] = {
    "window_title": {"zh": "CopyVocal 人声校正", "en": "CopyVocal Vocal Correction"},

    "nav_correction": {"zh": "校正", "en": "Correction"},
    "nav_results": {"zh": "结果", "en": "Results"},
    "nav_settings": {"zh": "设置", "en": "Settings"},

    "card_recordings": {"zh": "录音文件", "en": "Recordings"},
    "card_axes": {"zh": "校正方式", "en": "Correction"},
    "card_options": {"zh": "运行选项", "en": "Run options"},
    "card_progress": {"zh": "运行", "en": "Run"},
    "card_results": {"zh": "结果", "en": "Results"},
    "card_appearance": {"zh": "外观", "en": "Appearance"},
    "card_about": {"zh": "关于", "en": "About"},
    "expander_run_modes": {"zh": "更多运行方式", "en": "More run modes"},
    "expander_advanced": {"zh": "分析参数（通常无需修改）", "en": "Analysis parameters (usually unchanged)"},
    "expander_command": {"zh": "等价的命令行", "en": "Equivalent command line"},
    "expander_log": {"zh": "运行日志", "en": "Run log"},
    "switch_on": {"zh": "开", "en": "On"},
    "switch_off": {"zh": "关", "en": "Off"},

    "label_teacher": {"zh": "教师录音（目标）", "en": "Teacher recording (target)"},
    "label_student": {"zh": "学生录音（待校正）", "en": "Student recording (to correct)"},
    "label_output": {"zh": "输出目录", "en": "Output directory"},
    "label_teacher_short": {"zh": "教师", "en": "Teacher"},
    "label_student_short": {"zh": "学生", "en": "Student"},
    "button_browse": {"zh": "浏览…", "en": "Browse…"},
    "button_run": {"zh": "开始", "en": "Run"},
    "button_cancel": {"zh": "取消", "en": "Cancel"},
    "button_play": {"zh": "播放", "en": "Play"},
    "button_stop": {"zh": "停止", "en": "Stop"},
    "button_open_report": {"zh": "打开报告", "en": "Open report"},
    "button_open_output": {"zh": "打开输出文件夹", "en": "Open output folder"},
    "hint_inputs": {
        "zh": "把音频文件拖到窗口上，或用上面的按钮选择。输入应为分离后的、以单声部人声为主的录音；"
              "伴奏会干扰分析。",
        "en": "Drop audio files onto the window, or use the buttons above. Inputs should be isolated, "
              "predominantly monophonic vocals; accompaniment confuses the analysis.",
    },
    "label_lyric_backend": {"zh": "歌词识别后端", "en": "Lyric recognition backend"},
    "label_vocoder_device": {"zh": "声码器设备", "en": "Vocoder device"},
    "label_language": {"zh": "语言", "en": "Language"},
    "label_listening": {"zh": "试听", "en": "Listen"},
    "label_pitch_mode": {"zh": "改变音高（三选一）", "en": "Change pitch (one of three)"},
    "label_theme": {"zh": "主题", "en": "Theme"},
    "label_ui_language": {"zh": "界面语言", "en": "Interface language"},

    "tooltip_onnx": {
        "zh": "DirectML 使用 GPU 运行歌词模型；CPU 更慢但总能工作。",
        "en": "DirectML runs the lyric models on the GPU; CPU is slower but always works.",
    },
    "tooltip_vocoder": {
        "zh": "auto 在有 CUDA 时使用 GPU，否则使用 CPU。",
        "en": "auto uses the GPU when CUDA is available, otherwise the CPU.",
    },
    "tooltip_check_game": {
        "zh": "用 GAME 额外提取音符，写入 MIDI 供检视；不影响校正结果。",
        "en": "Extract notes with GAME as an extra MIDI artifact for inspection; it does not affect the correction.",
    },
    "tooltip_check_experiments": {
        "zh": "额外输出三个对照文件，便于判断每一步改动的影响。",
        "en": "Also render three A/B files, so the effect of each step can be judged separately.",
    },
    "tooltip_check_verify": {
        "zh": "对导出音频重新测量音高与长度，并写入 verification.json。",
        "en": "Remeasure the exported audio's pitch and length into verification.json.",
    },
    "tooltip_check_verify_timing": {
        "zh": "对输出音频再跑一次歌词识别与强制对齐，用来核对词边界；耗时较长。",
        "en": "Run lyric recognition and forced alignment again over the output to check word boundaries; slow.",
    },
    "tooltip_check_analyze_only": {
        "zh": "只生成分析文件，不合成音频。",
        "en": "Produce the analysis files only; no audio is synthesized.",
    },
    "tooltip_check_skip_analysis": {
        "zh": "直接使用输出目录里的分析产物。它不会因为分析设置或算法变化而重跑分析。",
        "en": "Reuse the analysis already in the output directory. It does not rerun analysis for changed settings or algorithms.",
    },
    "tooltip_check_fallback": {
        "zh": "允许在没有歌词对应的有声区段使用纯声学（活动度）对齐，结果可信度较低。",
        "en": "Allow activity-only acoustic timing where no lyric correspondence exists; the result is less trustworthy.",
    },
    "tooltip_check_timing": {
        "zh": "把学生的时间轴对齐到教师的词与音符边界。\n不勾选则保留学生自己的时间轴，只改音高。",
        "en": "Warp the student's timeline onto the teacher's word and note boundaries.\n"
              "Off keeps the student's own timeline and corrects pitch only.",
    },
    "tooltip_pitch_mode": {
        "zh": "关闭：音高完全按学生原样保留，一个采样都不动。\n"
              "拉到教师音高：把学生整体移到教师的音区，用实测的音区差作为一个常数；"
              "唱得比教师低或高时不会被留在本嗓，而是被拽到教师的高度。\n"
              "适应模式：逐词对到教师的音级，但整个旋律仍落在学生自己的音区，"
              "所以既能纠错音，又不会被拽出本嗓。勾选曲线时改为整条教师曲线平移到学生音区。",
        "en": "Off: the pitch is left exactly as the student sang it, frame for frame.\n"
              "Teacher: move the result into the teacher's register at the teacher's own levels, "
              "using the measured register difference as one global constant.\n"
              "Adaptive: put each word on the teacher's note level while the melody as a whole stays in "
              "the student's own register. With the contour checked, the whole teacher contour is "
              "transposed into the student's register instead.",
    },
    "tooltip_check_pitch_map": {
        "zh": "勾选后使用教师的连续音高曲线，滑音、颤音与音间过渡都来自教师。\n"
              "不勾选则保留学生自己的音高走向。\n"
              "「拉到教师音高」：不勾=学生曲线落到教师音区，勾=教师曲线与教师音区。\n"
              "「适应模式」：不勾=学生曲线逐词对到教师音级，勾=整条教师曲线平移到学生音区。",
        "en": "On uses the teacher's continuous pitch, so slides, vibrato and note choices are the teacher's.\n"
              "Off keeps the student's own pitch movement.\n"
              "Teacher mode: off places the student contour in the teacher's register, on uses the teacher's "
              "contour and register.\n"
              "Adaptive mode: off places each word on the teacher's note level, on transposes the whole "
              "teacher contour into the student's register.",
    },
    "tooltip_max_chunk": {
        "zh": "限制每个分析分块的长度，包含上下文。",
        "en": "Bound each analysis chunk, including context.",
    },
    "tooltip_context": {
        "zh": "为彼此不相连的分块保留上下文。",
        "en": "Preserve context around disjoint chunk cores.",
    },
    "tooltip_quality": {
        "zh": "低于该置信度的对应关系不参与正常合成。",
        "en": "Reject low-confidence correspondence for normal synthesis.",
    },
    "tooltip_stretch": {
        "zh": "整词时长比值超过该倍数（或低于其倒数）时判定为不可支持。",
        "en": "Reject whole-word duration ratios beyond this factor or below its reciprocal.",
    },
    "tooltip_floor": {
        "zh": "声码器做自然对数梅尔变换前的正幅度下限。",
        "en": "Set the positive amplitude floor before the vocoder's natural-log mel conversion.",
    },

    "check_game": {"zh": "提取音符（可选 MIDI 分析）", "en": "Extract notes (optional MIDI analysis)"},
    "check_experiments": {
        "zh": "输出 A/B 对照文件",
        "en": "Render A/B comparison files",
    },
    "check_verify": {"zh": "校验导出波形", "en": "Verify the exported waveform"},
    "check_verify_timing": {
        "zh": "重新测量输出词边界",
        "en": "Remeasure output word boundaries",
    },
    "check_analyze_only": {"zh": "只做分析（不合成）", "en": "Analyze only (no synthesis)"},
    "check_skip_analysis": {
        "zh": "复用输出目录中已有的分析结果",
        "en": "Reuse analysis already in the output directory",
    },
    "check_fallback": {
        "zh": "允许低置信度的声学兜底对齐",
        "en": "Allow low-confidence acoustic fallback timing",
    },
    "check_timing": {
        "zh": "改变节奏",
        "en": "Change timing",
    },
    "check_pitch_map": {
        "zh": "改变音高曲线",
        "en": "Change pitch contour",
    },
    "axis_hint": {
        "zh": "三项互相独立。音高关闭时完全不动音高，只改节奏。",
        "en": "The three are independent. With pitch off, nothing moves the pitch at all and only the timing changes.",
    },
    "pitch_mode_off": {"zh": "关闭", "en": "Off"},
    "pitch_mode_teacher": {"zh": "拉到教师音高", "en": "Match teacher pitch"},
    "pitch_mode_adaptive": {"zh": "适应模式", "en": "Adaptive"},

    "advanced_max_chunk": {"zh": "单块最大长度", "en": "Maximum chunk length"},
    "advanced_context": {"zh": "分块上下文长度", "en": "Chunk context length"},
    "advanced_quality": {"zh": "最低对应质量", "en": "Minimum correspondence quality"},
    "advanced_stretch": {"zh": "最大拉伸倍数", "en": "Maximum stretch factor"},
    "advanced_floor": {"zh": "声码器对数下限", "en": "Vocoder log floor"},

    "tab_log": {"zh": "日志", "en": "Log"},
    "tab_results": {"zh": "结果", "en": "Results"},
    "table_metrics": {"zh": "指标", "en": "Metrics"},
    "table_regions": {"zh": "对应关系", "en": "Correspondence"},
    "table_warnings": {"zh": "警告", "en": "Warnings"},
    "header_measurement": {"zh": "测量项", "en": "Measurement"},
    "header_value": {"zh": "数值", "en": "Value"},
    "headers_regions": {
        "zh": "教师,学生,状态,处理方式,质量,教师起点,教师终点,原因",
        "en": "Teacher,Student,Status,Handling,Quality,Teacher start,Teacher end,Reason",
    },
    "log_placeholder": {
        "zh": "任务运行时，管线输出会显示在这里。",
        "en": "Pipeline output appears here while a job runs.",
    },
    "warning_placeholder": {
        "zh": "管线给出的提示与限制会显示在这里。",
        "en": "Pipeline notes and limitations appear here.",
    },
    "detail_no_job": {"zh": "本次会话还没有开始任务。", "en": "No job has been started in this session."},
    "no_warnings": {"zh": "该输出目录没有记录警告。", "en": "This output directory records no warnings."},

    "summary_no_job": {"zh": "尚无任务产物", "en": "No job artifacts yet"},
    "summary_no_job_note": {
        "zh": "先运行一个任务，或把输出目录指向已有结果来查看它。",
        "en": "Run a job first, or point the output directory at existing results to review them.",
    },
    "summary_corrected_prefix": {"zh": "校正后的人声：", "en": "Corrected vocal: "},
    "summary_disclaimer": {
        "zh": "计划覆盖范围与实测波形质量是两件事，二者都不代表通过了听感验收。",
        "en": "Planned coverage and measured waveform quality are separate things, and neither is a "
              "listening-quality pass.",
    },

    "status_completed": {"zh": "已完成", "en": "Completed"},
    "status_completed_note": {
        "zh": "所有计划区段都由有支持的对应关系合成。",
        "en": "Every planned region was synthesized from supported correspondence.",
    },
    "status_unresolved": {"zh": "已完成，但存在未解决区段", "en": "Completed with unresolved regions"},
    "status_unresolved_note": {
        "zh": "部分内容被静音、模糊或未匹配。信任结果前请先检查对应关系与警告。",
        "en": "Some material was muted, ambiguous or unmatched. Inspect the correspondence and warnings "
              "before trusting the result.",
    },
    "status_analysis_only": {"zh": "分析完成", "en": "Analysis complete"},
    "status_analysis_only_note": {
        "zh": "只输出了检视用产物，没有合成音频。",
        "en": "Only inspection artifacts were written; no audio was synthesized.",
    },
    "status_failed": {"zh": "失败", "en": "Failed"},
    "status_failed_note": {
        "zh": "管线在生成校正人声之前就中止了。",
        "en": "The pipeline stopped before producing a corrected vocal.",
    },
    "status_cancelled": {"zh": "已取消", "en": "Cancelled"},
    "status_cancelled_note": {
        "zh": "任务被停止；已完成的分析缓存仍可复用。",
        "en": "The job was stopped; completed analysis caches remain reusable.",
    },
    "status_running": {"zh": "运行中", "en": "Running"},
    "status_running_note": {
        "zh": "界面显示的是进行中的状态；已有产物可能来自更早的一次运行。",
        "en": "This is the in-progress state; existing artifacts may come from an earlier run.",
    },
    "status_unknown": {"zh": "未知", "en": "Unknown"},
    "status_no_manifest_with_artifacts": {
        "zh": "无任务清单，但存在产物",
        "en": "No job manifest, but artifacts exist",
    },
    "status_no_manifest_note": {
        "zh": "该目录没有 job.json，因此无法给出这次任务的状态；下面的指标、对应关系与警告来自目录中实际存在的文件。",
        "en": "This directory has no job.json, so the job status is unavailable; the metrics, "
              "correspondence and warnings below come from the files that are actually present.",
    },
    "status_unknown_note": {"zh": "状态异常，请检查 job.json。", "en": "Unexpected status; check job.json."},

    "run_status_succeeded": {"zh": "已完成", "en": "Completed"},
    "run_status_failed": {"zh": "失败", "en": "Failed"},
    "run_status_cancelled": {"zh": "已取消", "en": "Cancelled"},

    "row_lyric_units": {"zh": "歌词单元", "en": "Lyric units"},
    "row_lyric_units_value": {"zh": "教师 {teacher} · 学生 {student}", "en": "Teacher {teacher} · Student {student}"},
    "row_unit_status": {"zh": "单元状态", "en": "Unit status"},
    "row_planned_coverage": {"zh": "计划歌词覆盖率", "en": "Planned lyric coverage"},
    "row_semantic_match": {"zh": "精确语义匹配率", "en": "Exact semantic match rate"},
    "row_unsupported_regions": {"zh": "未支持区段", "en": "Unsupported regions"},
    "row_ambiguous_matches": {"zh": "模糊匹配", "en": "Ambiguous matches"},
    "row_teacher_voiced_coverage": {"zh": "教师有声语义覆盖率", "en": "Teacher-voiced semantic coverage"},
    "row_exported_length": {"zh": "导出长度", "en": "Exported length"},
    "row_length_exact": {"zh": "精确", "en": "exact"},
    "row_length_mismatch": {"zh": "不一致", "en": "mismatch"},
    "row_length_value": {"zh": "{count} 采样（{state}）", "en": "{count} samples ({state})"},

    "pitch_vs_teacher": {"zh": "对比教师", "en": "vs teacher"},
    "pitch_vs_teacher_register": {"zh": "对比教师（换算到学生音区）", "en": "vs teacher (student register)"},
    "pitch_vs_target": {"zh": "对比渲染目标", "en": "vs render target"},
    "pitch_row_error": {"zh": "音高误差（{label}）", "en": "Pitch error ({label})"},
    "pitch_row_recall": {"zh": "有声召回率（{label}）", "en": "Voicing recall ({label})"},
    "pitch_row_value": {
        "zh": "中位 {median}，p95 {p95}，基于 {frames} 个共同有声帧",
        "en": "median {median}, p95 {p95} over {frames} joint voiced frames",
    },

    "timing_row_compared": {"zh": "可靠比对的单元", "en": "Reliably compared units"},
    "timing_row_coverage": {"zh": "全部教师单元中落在容差内", "en": "Teacher units within tolerance"},
    "timing_row_onset": {"zh": "起始误差", "en": "Onset error"},
    "timing_row_offset": {"zh": "结束误差", "en": "Offset error"},
    "timing_row_compared_value": {"zh": "{reliable} / {total}", "en": "{reliable} / {total}"},
    "timing_row_error_value": {"zh": "均值 {mean} · p95 {p95}", "en": "mean {mean} · p95 {p95}"},

    "row_synthesis_device": {"zh": "合成设备", "en": "Synthesis device"},
    "row_rendered_ratio": {"zh": "已渲染样本比例", "en": "Rendered sample share"},
    "row_voiced_ratio": {"zh": "有声帧比例", "en": "Voiced frame share"},
    "row_timing_changed": {"zh": "是否改变节奏", "en": "Timing changed"},
    "row_pitch_policy": {"zh": "音高策略", "en": "Pitch policy"},
    "row_register_offset": {"zh": "音区差（按持续音测量）", "en": "Register offset (sustained-note measured)"},
    "row_note_shifts": {"zh": "逐词移动的词数", "en": "Words moved individually"},
    "row_note_shifts_value": {"zh": "{moved} / {measured} 个", "en": "{moved} / {measured}"},
    "row_muted_regions": {"zh": "静音区段", "en": "Muted regions"},
    "row_acoustic_transitions": {"zh": "声学过渡", "en": "Acoustic transitions"},
    "row_peak_memory": {"zh": "峰值显存", "en": "Peak GPU memory"},
    "row_job_time": {"zh": "任务耗时", "en": "Job time"},
    "value_yes": {"zh": "是", "en": "yes"},
    "value_no": {"zh": "否", "en": "no"},
    "value_semitones": {"zh": "{value:+.2f} 半音", "en": "{value:+.2f} semitones"},
    "value_seconds": {"zh": "{value:.3f} 秒", "en": "{value:.3f} s"},

    "warning_limit_prefix": {"zh": "测量限制：", "en": "Measurement limit: "},
    "warning_structural_failure": {
        "zh": "导出波形未通过结构校验。",
        "en": "The exported waveform failed structural verification.",
    },

    "policy_none": {"zh": "不改变音高（学生原样）", "en": "Pitch unchanged (student as sung)"},
    "policy_student_contour_student_register": {
        "zh": "学生曲线 · 逐词对到教师音级 · 学生音区（适应模式）",
        "en": "Student contour · per-word teacher note level · student register (adaptive)",
    },
    "policy_teacher_contour_student_register": {
        "zh": "教师曲线 · 学生音区",
        "en": "Teacher contour · student register",
    },
    "policy_student_contour_teacher_register": {
        "zh": "学生曲线 · 教师音区",
        "en": "Student contour · teacher register",
    },
    "policy_teacher_contour_teacher_register": {
        "zh": "教师曲线 · 教师音区（绝对音高）",
        "en": "Teacher contour · teacher register (absolute pitch)",
    },

    "region_status_matched": {"zh": "匹配", "en": "matched"},
    "region_status_homophone": {"zh": "同音字", "en": "homophone"},
    "region_status_substitution": {"zh": "疑似替换", "en": "suspected substitution"},
    "region_status_missing": {"zh": "缺失", "en": "missing"},
    "region_status_unresolved": {"zh": "未解决", "en": "unresolved"},
    "region_status_unvoiced_gap": {"zh": "无声音隙", "en": "unvoiced gap"},
    "region_status_phonetic_only": {"zh": "仅发音匹配", "en": "phonetic only"},
    "region_status_extra": {"zh": "多余", "en": "extra"},
    "region_status_ambiguous": {"zh": "模糊", "en": "ambiguous"},

    "region_mode_synthesize": {"zh": "合成", "en": "synthesize"},
    "region_mode_mute": {"zh": "静音", "en": "mute"},
    "region_mode_silence": {"zh": "静音", "en": "silence"},
    "region_mode_acoustic_fallback": {"zh": "声学兜底", "en": "acoustic fallback"},

    "region_reason_voiced_teacher_region_without_lyric_correspondence": {
        "zh": "教师有声区段没有歌词对应",
        "en": "voiced teacher region without lyric correspondence",
    },
    "region_reason_teacher_lyrics_unavailable": {
        "zh": "教师歌词不可用",
        "en": "teacher lyrics unavailable",
    },
    "region_reason_untranscribed_activity_no_lyric_correspondence": {
        "zh": "未转写区段没有歌词对应",
        "en": "untranscribed activity without lyric correspondence",
    },
    "region_reason_pitch_unvoiced_not_acoustic_silence": {
        "zh": "无音高不等于静音",
        "en": "pitch-unvoiced is not acoustic silence",
    },
    "region_reason_multiple_lexically_equivalent_paths_progress_is_only_weak_evidence": {
        "zh": "存在多条等价路径，仅凭进度无法判定",
        "en": "multiple lexically equivalent paths; progress is only weak evidence",
    },
    "region_reason_consonant_limits_infeasible": {
        "zh": "辅音时长限制下无法排布",
        "en": "infeasible under the consonant duration limits",
    },
    "region_reason_no_phone_timeline_in_window": {
        "zh": "该窗口没有音素时间轴",
        "en": "no phone timeline in this window",
    },
    "region_reason_no_register_offset": {"zh": "没有可用音区差", "en": "no register offset available"},
    "region_reason_already_at_target_level": {"zh": "已在目标音级", "en": "already at the target level"},
    "region_reason_no_teacher_voiced_level": {
        "zh": "教师该区段无有声音高",
        "en": "no teacher voiced level in this region",
    },
    "region_reason_no_student_voiced_level": {
        "zh": "学生该区段无有声音高",
        "en": "no student voiced level in this region",
    },
    "region_reason_teacher_contour_used": {
        "zh": "使用教师曲线，不逐词位移",
        "en": "teacher contour used; no per-word shift",
    },
    "region_reason_matched": {"zh": "正常匹配", "en": "matched normally"},
    "region_reason_homophone": {"zh": "同音字", "en": "homophone"},
    "region_reason_possible_lyric_mismatch_or_asr_error": {
        "zh": "可能歌词不一致或识别有误",
        "en": "possible lyric mismatch or ASR error",
    },
    "region_reason_no_corresponding_student_lyric_unit": {
        "zh": "没有对应的学生歌词单元",
        "en": "no corresponding student lyric unit",
    },

    "audio_teacher": {"zh": "教师参考", "en": "Teacher reference"},
    "audio_student": {"zh": "学生原始", "en": "Student original"},
    "audio_corrected": {"zh": "校正后学生", "en": "Corrected student"},
    "audio_identity": {"zh": "identity 重建", "en": "Identity reconstruction"},
    "audio_pitch_only": {"zh": "仅改音高", "en": "Pitch only"},
    "audio_time_only": {"zh": "仅改节奏", "en": "Timing only"},

    "stage_analysis": {"zh": "分析：歌词、音素、音高与音符", "en": "Analysis: lyrics, phonemes, pitch and notes"},
    "stage_alignment": {
        "zh": "对应：匹配歌词单元并建立时间映射",
        "en": "Correspondence: match lyric units and build time maps",
    },
    "stage_render": {
        "zh": "合成：变换特征并渲染校正后的人声",
        "en": "Synthesis: warp features and render the corrected vocal",
    },
    "stage_verification": {"zh": "校验：重新测量导出的波形", "en": "Verification: remeasure the exported waveform"},
    "stage_timing_verification": {
        "zh": "时间核对：重新识别输出词边界",
        "en": "Timing check: remeasure output word boundaries",
    },
    "stage_review": {"zh": "汇总：整理音频、警告与来源信息", "en": "Review: gather audio, warnings and provenance"},

    "track_teacher": {"zh": "教师", "en": "teacher"},
    "track_student": {"zh": "学生", "en": "student"},
    "phase_lyrics": {"zh": "歌词", "en": "Lyrics"},
    "phase_phonemes": {"zh": "音素", "en": "Phonemes"},
    "phase_notes": {"zh": "音符", "en": "Notes"},
    "detail_chunk": {"zh": "{phase}：{track} 第 {index}/{total} 块", "en": "{phase}: {track} chunk {index}/{total}"},
    "detail_pitch": {"zh": "提取连续音高：{track}", "en": "Extracting continuous pitch: {track}"},
    "detail_render": {"zh": "正在渲染：{mode}", "en": "Rendering: {mode}"},
    "detail_loading_vocoder": {"zh": "正在加载声码器（{device}）", "en": "Loading the vocoder ({device})"},
    "detail_starting": {"zh": "正在启动管线。", "en": "Starting the pipeline."},
    "detail_running": {"zh": "运行中…管线输出显示在日志页。", "en": "Running… pipeline output appears on the Log tab."},
    "detail_finished": {"zh": "已结束：{status}，耗时 {seconds:.1f} 秒。", "en": "Finished: {status} after {seconds:.1f} s."},
    "detail_cancelled": {"zh": "已取消。已完成的分析缓存仍可复用。", "en": "Cancelled. Completed analysis caches remain reusable."},
    "detail_cancelling": {"zh": "正在取消，等待各工作进程停止。", "en": "Cancelling; waiting for the worker processes to stop."},

    "statusbar_ready": {"zh": "就绪。", "en": "Ready."},
    "statusbar_running": {"zh": "运行中。", "en": "Running."},
    "statusbar_job": {"zh": "任务{status}。", "en": "Job {status}."},
    "statusbar_playback_unavailable": {"zh": "无法播放：{message}", "en": "Playback unavailable: {message}"},
    "statusbar_prefs_failed": {"zh": "无法保存界面偏好设置。", "en": "Could not save the interface preferences."},

    "dialog_cannot_start": {"zh": "无法开始", "en": "Cannot start"},
    "dialog_pipeline_failed": {"zh": "管线失败", "en": "Pipeline failed"},
    "dialog_job_running": {"zh": "有任务正在运行", "en": "A job is running"},
    "dialog_reuse_title": {"zh": "复用已有分析", "en": "Reuse existing analysis"},
    "dialog_reuse_body": {
        "zh": "复用已有分析不会因为分析设置或算法变化而重跑分析，合成阶段仍会重新读取输入音频。\n\n是否继续？",
        "en": "Reusing the analysis does not rerun it for changed analysis settings or algorithms, and "
              "synthesis still reloads the input audio.\n\nContinue?",
    },
    "dialog_job_running_body": {"zh": "停止正在运行的任务并关闭窗口？", "en": "Stop the running job and close the window?"},
    "dialog_failed_fallback": {"zh": "管线在合成之前就中止了。", "en": "The pipeline stopped before synthesis."},
    "dialog_failed_body": {
        "zh": "{error}\n\n日志页与输出目录中的 processing.log 记录了管线的输出。",
        "en": "{error}\n\nThe log tab and processing.log in the output directory record the pipeline output.",
    },
    "bullet_item": {"zh": "• {item}", "en": "• {item}"},

    "validate_choose_recording": {"zh": "请选择{label}录音。", "en": "Choose the {label} recording."},
    "validate_file_not_found": {"zh": "{label}文件不存在：{path}", "en": "{label} file not found: {path}"},
    "validate_same_file": {
        "zh": "教师录音与学生录音不能是同一个文件。",
        "en": "The teacher and student recordings cannot be the same file.",
    },
    "validate_choose_output": {"zh": "请选择输出目录。", "en": "Choose an output directory."},
    "validate_output_is_file": {
        "zh": "输出路径已存在且是文件：{path}",
        "en": "The output path exists and is a file: {path}",
    },
    "validate_conflict": {
        "zh": "「只做分析」与「复用已有分析」不能同时启用，请取消其中一个。",
        "en": "Analyze only and reuse analysis cannot both be enabled; clear one of them.",
    },
    "validate_missing_analysis": {
        "zh": "复用已有分析需要输出目录中存在：{files}。",
        "en": "Reusing analysis requires these files in the output directory: {files}.",
    },
    "validate_range": {
        "zh": "{name}必须大于 {low} 且不超过 {high}。",
        "en": "{name} must be greater than {low} and at most {high}.",
    },

    "theme_light": {"zh": "浅色", "en": "Light"},
    "theme_dark": {"zh": "深色", "en": "Dark"},
    "theme_auto": {"zh": "跟随系统", "en": "Follow system"},
    "language_zh": {"zh": "中文", "en": "中文"},
    "language_en": {"zh": "English", "en": "English"},
    "version_info": {"zh": "CopyVocal {version}", "en": "CopyVocal {version}"},
    "about_body": {
        "zh": "把学生的人声录音按教师的节奏与音高重新校正。界面只驱动命令行管线，本身不做推理。",
        "en": "Retimes a student's vocal to the teacher's timing and pitch. The interface only drives the "
              "command-line pipeline and adds no inference of its own.",
    },
}


def set_language(language: str) -> None:
    """Switch the active language; anything but English selects Chinese."""
    global current_language
    value = str(language).strip().lower()
    current_language = "en" if value in {"en", "english"} else "zh"


def tr(key: str, **kwargs) -> str:
    """Return the text for one key, falling back to Chinese and then to the key."""
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(current_language) or entry.get("zh") or key
    if kwargs:
        return text.format(**kwargs)
    return text
