from __future__ import annotations

import re

from gui.i18n import tr


def stage_labels() -> tuple[tuple[str, str], ...]:
    """Stage keys with their labels in the active language."""
    return tuple(
        (key, tr(f"stage_{key}"))
        for key in ("analysis", "alignment", "render", "verification", "timing_verification", "review")
    )


def stage_track_labels() -> dict[str, str]:
    return {"teacher": tr("track_teacher"), "student": tr("track_student")}


def phase_labels() -> dict[str, str]:
    return {"Lyrics": tr("phase_lyrics"), "Phonemes": tr("phase_phonemes"), "Notes": tr("phase_notes")}


def pitch_mode_labels() -> tuple[str, ...]:
    return (tr("pitch_mode_off"), tr("pitch_mode_teacher"), tr("pitch_mode_adaptive"))


PITCH_MODE_VALUES = ("off", "teacher", "adaptive")


def status_summary(status: str) -> tuple[str, str]:
    """Headline and note for one job status, in the active language."""
    key = {
        "completed": "status_completed",
        "completed_with_unresolved": "status_unresolved",
        "analysis_complete": "status_analysis_only",
        "failed": "status_failed",
        "cancelled": "status_cancelled",
        "running": "status_running",
    }.get(status)
    if key is None:
        return tr("status_unknown"), tr("status_unknown_note")
    return tr(key), tr(f"{key}_note")


def run_status_headline(status: str) -> str:
    return tr(f"run_status_{status}") if status in {"succeeded", "failed", "cancelled"} else status


EXACT_WARNINGS = {
    "Audio is near silence; lyric analysis was not attempted.":
        "音频接近静音，未进行歌词分析。",
    "No reliable lyric spans are available; lyric correction must not be reported as successful.":
        "没有可靠的歌词片段，不能把这次歌词校正报告为成功。",
    "No timed teacher lyric units were available.":
        "没有带时间信息的教师歌词单元。",
    "No timed student lyric units were available.":
        "没有带时间信息的学生歌词单元。",
    "No voiced F0 frames remain after voicing and correspondence checks.":
        "经过有声判定与对应关系检查后，没有剩余有声基频帧。",
    "No renderer target pitch was supplied; target execution and render support cannot be audited.":
        "没有提供渲染目标音高，无法审核目标执行情况与渲染支持范围。",
    "No reliable output word boundaries can be compared; timing accuracy is unavailable.":
        "没有可靠的输出词边界可用于比对，无法给出时间准确度。",
    "Overlapping word boundaries were clipped and their confidence reduced.":
        "存在重叠的词边界，已裁剪并降低其置信度。",
    "Word order was normalized by time; inspect overlapping ASR timestamps.":
        "词序按时间重新排序，请检查重叠的识别时间戳。",
    "Multi-unit ASR words use estimated internal boundaries; re-align them for precise editing.":
        "包含多个单元的识别结果使用估计的内部边界；需要精确编辑时请重新对齐。",
    "Untranscribed teacher voicing lacks corresponding student material and is muted.":
        "教师未转写的有声区段没有对应的学生素材，已静音。",
    "Untranscribed voiced gaps have activity-only timing with no semantic validation.":
        "未转写的有声间隙只按活动度对齐，没有语义验证。",
    "All timed teacher lyric material is muted because correspondence is unsupported; this is not a successful correction.":
        "教师的全部歌词内容都没有可支持的对应关系，已静音；这不算一次成功的校正。",
    "Output sample count or sample rate differs from the teacher reference.":
        "输出采样数或采样率与教师参考不一致。",
    "Output reaches digital full scale; inspect the reported full-scale sample count for clipping.":
        "输出已达到数字满刻度；请查看报告的满刻度样本数，确认是否削波。",
    "Output channels show phase cancellation; pitch measured on the strongest channel.":
        "输出声道之间存在相位抵消；音高改在能量最强的声道上测量。",
    "Pitch verification was skipped because the waveform is empty or non-finite.":
        "波形为空或包含非有限值，已跳过音高校验。",
    "Teacher has no pitch frames; teacher accuracy metrics are unavailable.":
        "教师没有音高帧，无法给出以教师为参考的准确度指标。",
    "Some teacher voiced frames lack semantic support; full reference metrics retain them in the denominator.":
        "部分教师有声帧没有语义支持；整体参考指标仍把它们计入分母。",
    "Teacher or output lyric analysis is incomplete; valid completed chunks may still contribute measured spans.":
        "教师或输出的歌词分析不完整；已完成的可用分块仍会贡献测量结果。",
    "Output waveform verification was skipped for this job.":
        "本次任务跳过了输出波形校验。",
    "Pitch-only reconstruction preserves student pitch outside supported correspondences.":
        "「仅改音高」实验在有支持的对应关系之外保留学生自己的音高。",
    "Some TiFA boundaries include at least 30 ms of exact output silence; these are alignment estimates, not measured articulation onsets.":
        "部分强制对齐边界包含至少 30 毫秒的输出精确静音；这些是对齐估计，不是实测的发音起点。",
    "Timing errors cover reliable exact lexical matches only; uncovered teacher units remain in the overall coverage denominator.":
        "时间误差只统计可靠的精确词匹配；未覆盖的教师单元仍计入整体分母。",
    "Transcript differences can be ASR errors or rendered lyric problems; they do not prove omitted, added, or changed singing.":
        "转写差异可能来自识别错误，也可能来自渲染后的歌词问题；它不能证明漏唱、多唱或改词。",
    "The output keeps the student's timeline, so it is not sample-aligned with the teacher and the teacher-timeline pitch and coverage metrics are not measured.":
        "本次输出保留学生自己的时间轴，因此与教师并非逐采样对齐；以教师时间轴为准的音高与覆盖指标不参与测量。",
    "Timing accuracy against the teacher is only defined when the timing was corrected; this job did not change the timing.":
        "只有在校正过节奏时，与教师的时间准确度才有意义；本次任务没有改变节奏。",
    "Output word timing was not compared with the teacher: this render keeps the student's timeline, so teacher boundaries do not describe the same moments.":
        "没有把输出的词时间与教师比对：本次输出保留学生的时间轴，教师边界并不描述同一时刻。",
    "Output sample count or sample rate differs from the student recording.":
        "输出采样数或采样率与学生录音不一致。",
    "Not measured: the planned boundaries are teacher times and this output keeps the student's timeline.":
        "未测量：计划边界是教师时间，而本次输出保留学生的时间轴。",
    "Sorted selection must match reference timestamps.":
        "比对范围必须与参考时间戳一一对应。",
}

EXACT_JOB_ERRORS = {
    "Pitch analysis is missing; correction cannot run.": "缺少音高分析，无法进行校正。",
    "No supported vocal correspondences; inspect analysis and alignment artifacts.":
        "没有可支持的人声对应关系；请检查分析与对应关系产物。",
    "Rendered waveform failed structural verification; inspect verification.json.":
        "渲染出的波形未通过结构校验；请检查 verification.json。",
}

TEMPLATE_JOB_ERRORS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^(\w+)\.py failed with exit code (-?\d+)\. See (.+)$"),
     "{0}.py 以退出码 {1} 结束，详见 {2}"),
    (re.compile(r"^(\w+) analysis does not match the current input file\.$"),
     "{0}的分析结果与当前输入文件不匹配。"),
    (re.compile(r"^(\w+) analysis language does not match (\w+); rerun without --skip-analysis\.$"),
     "{0}的分析语言与 {1} 不一致；请去掉 --skip-analysis 重新运行。"),
    (re.compile(r"^(\w+) analysis lacks the Japanese pronunciation pipeline; rerun without --skip-analysis\.$"),
     "{0} 的分析缺少日文发音管线；请去掉 --skip-analysis 重新运行。"),
    (re.compile(r"^Unsupported language: (\w+)\. Choose from (.+)\.$"),
     "不支持的语言：{0}，可选 {1}。"),
)


def job_error_text(sentence: str) -> str:
    """Translate a job-level failure for the dialog, or return it unchanged."""
    text = str(sentence)
    if text in EXACT_JOB_ERRORS:
        return EXACT_JOB_ERRORS[text]
    for pattern, template in TEMPLATE_JOB_ERRORS:
        match = pattern.match(text)
        if match:
            values = [stage_track_labels().get(value, value) for value in match.groups()]
            return template.format(*values)
    return text


EXACT_LIMITATIONS = {
    "RMVPE is also used for teacher analysis; agreement is not an independent proof of acoustic correctness.":
        "教师分析同样使用 RMVPE，因此一致性并不能独立证明声学正确性。",
    "Word and phoneme timings have not been remeasured in the rendered waveform; planned boundaries are not measured output boundaries.":
        "尚未在渲染波形中重新测量词与音素的时间；计划边界不等于实测的输出边界。",
    "No objective timbre-preservation score is claimed; listening to identity and corrected samples is required.":
        "本管线不给出客观的音色保真分数；必须试听 identity 与校正后样本自行判断。",
    "Pitch was mapped relative to the student's own register, so agreement with the teacher's absolute pitch measures the intended register offset, not an error. Use actual_vs_teacher_relative and actual_vs_render_target for the aimed pitch.":
        "音高是相对学生自己的音区映射的，因此与教师绝对音高的一致性衡量的是有意保留的音区差，而不是误差。"
        "请用 actual_vs_teacher_relative 与 actual_vs_render_target 查看实际瞄准的音高。",
    "A selected stable and energetic voiced subset only; YIN can choose an octave or a strong harmonic and does not validate words, consonants, or timbre.":
        "只统计了稳定且有能量的有声片段；YIN 可能选到八度或强谐波，也不验证歌词、辅音或音色。",
    "ASR and TiFA have recognition and boundary uncertainty; output measurements are model estimates, not human annotations.":
        "识别与强制对齐都存在不确定度；输出测量是模型估计，不是人工标注。",
    "Missing transcript units cannot establish whether the audio omitted a lyric or the recognizer failed.":
        "缺少转写单元时，无法判断是音频漏唱还是识别失败。",
    "Multi-unit words with estimated internal boundaries, low-quality spans, and clipped overlaps do not contribute precise timing metrics.":
        "含估计内部边界的多单元词、低质量片段与被裁剪的重叠部分都不计入精确时间指标。",
    "Only exact, structurally valid, unambiguous lexical matches contribute timing errors; homophones and suspected substitutions are reported separately.":
        "只有精确、结构有效且无歧义的词匹配才计入时间误差；同音字与疑似替换单独汇报。",
    "Exact-silence evidence only; boundaries and timing error statistics are unchanged.":
        "仅依据精确静音证据；边界与时间误差统计不受影响。",
    "Teacher and output use the same model family, so correlated alignment errors can remain.":
        "教师与输出使用同一系列模型，可能残留相关的对齐误差。",
    "Word timing does not measure pitch, timbre, intelligibility, or perceptual audio quality.":
        "词时间测量不衡量音高、音色、可懂度或听感质量。",
    "Reliable means lexical and structural checks passed, not a calibrated acoustic boundary confidence; TiFA can absorb silence or missing ASR words into an adjacent matched word.":
        "「可靠」指通过了词法与结构检查，并不等于经过校准的声学边界置信度；强制对齐可能吞掉静音或放错边界。",
    "Homophones and suspected substitutions are excluded from boundary errors because their identity is not established.":
        "同音字与疑似替换的身份未确认，因此不计入边界误差。",
    "Teacher-voiced semantic coverage counts planned correspondence, not measured output audio.":
        "教师有声语义覆盖率统计的是计划对应关系，不是实测的输出音频。",
    "Boundaries come from the alignment plan, not from output measurements.":
        "边界来自对齐计划，而不是输出实测。",
    "Some teacher units lack student material and are muted; they remain in the denominator.":
        "部分教师单元没有学生素材而被静音；它们仍计入分母。",
    "Every teacher unit is compared against the best available output candidate.":
        "每个教师单元都与最可用的输出候选项比对。",
    "Alignment quality is a structural score, not a calibrated model confidence.":
        "对齐质量是一个结构评分，不是经过校准的模型置信度。",
    "Output word boundaries were not remeasured for this job.":
        "本次任务没有重新测量输出词边界。",
    "No reliable exact word observations are available for timing errors.":
        "没有可靠的精确词观测可用于时间误差统计。",
    "Unvoiced intervals are omitted from this chart.":
        "图中略去了无声区间。",
}

TEMPLATE_WARNINGS: tuple[tuple[re.Pattern, str, tuple[str, ...]], ...] = (
    (re.compile(r"^Possible lyric mismatch or ASR error: (.+) versus (.+)\.$"),
     "歌词可能不一致或识别有误：教师{0} / 学生{1}。", ("teacher", "student")),
    (re.compile(r"^Unsupported stretch ratio ([-\d.]+) for teacher word (.+)\.$"),
     "教师词{0}需要 {1} 倍拉伸，超出支持范围。", ("word", "ratio")),
    (re.compile(r"^Extreme local phoneme stretch in teacher word (.+); inspect audio quality\.$"),
     "教师词{0}出现极端局部音素拉伸，请检查听感质量。", ("word",)),
    (re.compile(r"^Ambiguous repeated lyrics affect (\d+) matches; progress alone cannot identify the omitted repetition\.$"),
     "重复歌词使 {0} 处匹配变得模糊；仅凭进度无法确定被省略的是哪一遍。", ()),
    (re.compile(r"^Chunk boundary word overlap at ([\d.]+)s requires review\.$"),
     "分块边界处 {0} 秒存在词重叠，需要人工检查。", ()),
    (re.compile(r"^Chunk (\d+) has lyrics but no valid core word spans\.$"),
     "第 {0} 块有歌词，但没有有效的主体词片段。", ()),
    (re.compile(r"^ASR failed for chunk (\d+); retry analysis before correction\.$"),
     "第 {0} 块的歌词识别失败；请重新分析后再校正。", ()),
    (re.compile(r"^TiFA failed for chunk (\d+); retry analysis before correction\.$"),
     "第 {0} 块的强制对齐失败；请重新分析后再校正。", ()),
    (re.compile(r"^ASR did not provide (.+) lyrics\. Activity-only timing has no semantic validation\.$"),
     "识别没有提供{0}的歌词；仅按活动度对齐，没有语义验证。", ()),
    (re.compile(r"^(\d+) unsupported or low-quality regions remain unresolved; acoustic rendering does not establish lyric correspondence\.$"),
     "{0} 个未支持或低质量区段仍未解决；声学渲染并不能确立歌词对应关系。", ()),
    (re.compile(r"^No global register offset was applied: (.+)$"),
     "没有应用整体音区差：{0}", ()),
    (re.compile(r"^Only (\d+) matched region\(s\) carry voiced pitch on both sides; a register .*$"),
     "只有 {0} 个匹配区段在两侧都有有声音高，无法据此判断整体音区差（至少需要两个）；"
     "教师的旋律改为对齐到各区段自身的音高。", ()),
    (re.compile(r"^Measured register difference of (-?\d+) cents exceeds two octaves; .*$"),
     "测得音区差 {0} 音分，超过两个八度；这更可能是八度跟踪错误，因此不应用整体音区差，"
     "各匹配区段改为对齐到自身的音高。", ()),
    (re.compile(r"^The teacher's register is the target, so the teacher's absolute levels are used and no register offset is applied\.$"),
     "目标是教师的音区，因此使用教师的绝对音高，不应用音区差。", ()),
    (re.compile(r"^Pitch change is disabled; the student's own pitch is used unchanged\.$"),
     "已关闭音高改变，保留学生自己的音高。", ()),
    (re.compile(r"^Japanese romaji ASR failed for (.+) chunk (\d+)\.$"),
     "日文罗马字识别失败：{0} 第 {1} 块。", ()),
    (re.compile(r"^Register-relative pitch mapping used a (-?[\d.]+)-cent register offset; the teacher's absolute pitch is not the intended output pitch\.$"),
     "音高映射使用了 {0} 音分的整体音区差，因此教师绝对音高并不是本次想要输出的音高。", ()),
)


def _strip_quotes(value: str) -> str:
    """Drop the repr quotes the workers put around a word, then use Chinese ones."""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return f"“{text}”"


def warning_text(sentence: str) -> str:
    """Translate one pipeline sentence in Chinese, or return it unchanged.

    English is the pipeline's own language, so the English UI shows it verbatim."""
    from gui.i18n import current_language

    text = str(sentence)
    if current_language != "zh":
        return text
    if text in EXACT_WARNINGS:
        return EXACT_WARNINGS[text]
    if text in EXACT_LIMITATIONS:
        return EXACT_LIMITATIONS[text]
    for pattern, template, groups in TEMPLATE_WARNINGS:
        match = pattern.match(text)
        if match:
            values = list(match.groups())
            if groups == ("word", "ratio"):
                ratios, words = values
                return template.format(_strip_quotes(words), ratios)
            if groups == ("teacher", "student"):
                return template.format(*(_strip_quotes(value) for value in values))
            if groups == ("word",):
                return template.format(_strip_quotes(values[0]))
            return template.format(*values)
    return text


def warning_list(items) -> list[str]:
    """Translate every sentence in one list, preserving order."""
    return [warning_text(item) for item in items]


def status_label(kind: str) -> str:
    return tr(f"region_status_{kind}") if f"region_status_{kind}" in _KNOWN_REGION_STATUSES else str(kind)


def mode_label(kind: str) -> str:
    return tr(f"region_mode_{kind}") if f"region_mode_{kind}" in _KNOWN_REGION_MODES else str(kind)


def reason_label(kind: str) -> str:
    key = f"region_reason_{kind}"
    return tr(key) if key in _KNOWN_REGION_REASONS else str(kind)


def policy_label(policy: str) -> str:
    return tr(f"policy_{policy}") if f"policy_{policy}" in _KNOWN_POLICIES else str(policy)


def _known_keys(prefix: str) -> frozenset[str]:
    from gui.i18n import _STRINGS

    return frozenset(key for key in _STRINGS if key.startswith(prefix))


_KNOWN_REGION_STATUSES = _known_keys("region_status_")
_KNOWN_REGION_MODES = _known_keys("region_mode_")
_KNOWN_REGION_REASONS = _known_keys("region_reason_")
_KNOWN_POLICIES = _known_keys("policy_")
