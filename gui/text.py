"""Chinese interface text for the desktop front end.

All user-visible wording lives here as data, so the interface code stays in
English like the rest of the project and every string can be reviewed in one
place. Keys are English identifiers; values are the Simplified Chinese shown to
the user.

Pipeline artifacts (analysis, alignment, verification JSON) are written in
English by the workers. ``warning_text`` translates the sentences the interface
displays from those files, and deliberately returns anything unrecognized
unchanged: a warning the interface has not been taught about is exactly the one a
user must still be able to read.
"""

from __future__ import annotations

import re

# ── Window chrome ─────────────────────────────────────────────────
WINDOW_TITLE = "CopyVocal 人声校正"
GROUP_RECORDINGS = "录音文件"
GROUP_OPTIONS = "选项"
GROUP_ADVANCED = "高级"
GROUP_PROGRESS = "进度"
LABEL_TEACHER = "教师录音（目标）"
LABEL_STUDENT = "学生录音（待校正）"
LABEL_OUTPUT = "输出目录"
BUTTON_BROWSE = "浏览…"
BUTTON_RUN = "开始"
BUTTON_CANCEL = "取消"
BUTTON_PLAY = "播放"
BUTTON_STOP = "停止"
BUTTON_OPEN_REPORT = "打开报告"
BUTTON_OPEN_OUTPUT = "打开输出文件夹"
HINT_INPUTS = ("把音频文件拖到窗口上，或用上面的按钮选择。输入应为分离后的、以单声部人声为主的录音；"
               "伴奏会干扰分析。")
LABEL_LYRIC_BACKEND = "歌词识别后端"
LABEL_VOCODER_DEVICE = "声码器设备"
LABEL_LANGUAGE = "语言"
LABEL_LISTENING = "试听"

TOOLTIP_ONNX = "DirectML 使用 GPU 运行歌词模型；CPU 更慢但总能工作。"
TOOLTIP_VOCODER = "auto 在有 CUDA 时使用 GPU，否则使用 CPU。"

# ── Option checkboxes ─────────────────────────────────────────────
CHECK_GAME = "提取音符（可选 MIDI 分析）"
CHECK_EXPERIMENTS = "输出 A/B 对照（identity / 仅音高 / 仅节奏）"
CHECK_VERIFY = "校验导出波形"
CHECK_VERIFY_TIMING = "重新测量输出词边界（较慢的 ASR/TiFA 流程）"
CHECK_ANALYZE_ONLY = "只做分析（不合成）"
CHECK_SKIP_ANALYSIS = "复用输出目录中已有的分析结果"
CHECK_FALLBACK = "允许低置信度的声学兜底对齐"
CHECK_TIMING = "改变节奏（学生节奏 → 教师节奏）"
LABEL_PITCH_MODE = "音高："
PITCH_MODE_LABELS = ("关闭", "拉到教师音高", "适应模式")
PITCH_MODE_VALUES = ("off", "teacher", "adaptive")
CHECK_PITCH_MAP = "改变 pitch 曲线（使用教师的滑音与颤音）"
AXIS_HINT = ("三项互相独立。「音高」是关闭 / 拉到教师音高 / 适应模式的三选一，「改变 pitch 曲线」决定用谁的音高走向；"
             "音高关闭时完全不动音高，只改节奏。")

TOOLTIP_CHECK_GAME = "用 GAME 额外提取音符，写入 MIDI 供检视；不影响校正结果。"
TOOLTIP_CHECK_EXPERIMENTS = "额外输出三个对照文件，便于判断每一步改动的影响。"
TOOLTIP_CHECK_VERIFY = "对导出音频重新测量音高与长度，并写入 verification.json。"
TOOLTIP_CHECK_VERIFY_TIMING = "对输出音频再跑一次歌词识别与强制对齐，用来核对词边界；耗时较长。"
TOOLTIP_CHECK_ANALYZE_ONLY = "只生成分析文件，不合成音频。"
TOOLTIP_CHECK_SKIP_ANALYSIS = "直接使用输出目录里的分析产物。它不会因为分析设置或算法变化而重跑分析。"
TOOLTIP_CHECK_FALLBACK = "允许在没有歌词对应的有声区段使用纯声学（活动度）对齐，结果可信度较低。"
TOOLTIP_CHECK_TIMING = ("把学生的时间轴对齐到教师的词与音符边界。\n"
                        "不勾选则保留学生自己的时间轴，只改音高。")
TOOLTIP_PITCH_MODE = ("关闭：音高完全按学生原样保留，一个采样都不动。\n"
                      "拉到教师音高：把学生整体移到教师的音区，用实测的音区差作为一个常数；"
                      "唱得比教师低或高时不会被留在本嗓，而是被拽到教师的高度。\n"
                      "适应模式：逐词对到教师的音级，但整个旋律仍落在学生自己的音区，"
                      "所以既能纠错音，又不会被拽出本嗓。勾选曲线时改为整条教师曲线平移到学生音区。")
TOOLTIP_CHECK_PITCH_MAP = ("勾选后使用教师的连续音高曲线，滑音、颤音与音间过渡都来自教师。\n"
                           "不勾选则保留学生自己的音高走向。\n"
                           "「拉到教师音高」：不勾=学生曲线落到教师音区，勾=教师曲线与教师音区。\n"
                           "「适应模式」：不勾=学生曲线逐词对到教师音级，勾=整条教师曲线平移到学生音区。")

# ── Advanced group ────────────────────────────────────────────────
ADVANCED_MAX_CHUNK = "单块最大长度"
ADVANCED_CONTEXT = "分块上下文长度"
ADVANCED_QUALITY = "最低对应质量"
ADVANCED_STRETCH = "最大拉伸倍数"
ADVANCED_FLOOR = "声码器对数下限"
TOOLTIP_MAX_CHUNK = "限制每个分析分块的长度，包含上下文。"
TOOLTIP_CONTEXT = "为彼此不相连的分块保留上下文。"
TOOLTIP_QUALITY = "低于该置信度的对应关系不参与正常合成。"
TOOLTIP_STRETCH = "整词时长比值超过该倍数（或低于其倒数）时判定为不可支持。"
TOOLTIP_FLOOR = "声码器做自然对数梅尔变换前的正幅度下限。"

# ── Tabs, tables, results panel ───────────────────────────────────
TAB_LOG = "日志"
TAB_RESULTS = "结果"
TABLE_METRICS = "指标"
TABLE_REGIONS = "对应关系"
TABLE_WARNINGS = "警告"
HEADER_MEASUREMENT = "测量项"
HEADER_VALUE = "数值"
HEADERS_REGIONS = ("教师", "学生", "状态", "处理方式", "质量", "教师起点", "教师终点", "原因")
LOG_PLACEHOLDER = "任务运行时，管线输出会显示在这里。"
WARNING_PLACEHOLDER = "管线给出的提示与限制会显示在这里。"
DETAIL_NO_JOB = "本次会话还没有开始任务。"
NO_WARNINGS = "该输出目录没有记录警告。"

SUMMARY_NO_JOB = "尚无任务产物"
SUMMARY_NO_JOB_NOTE = "先运行一个任务，或把输出目录指向已有结果来查看它。"
SUMMARY_CORRECTED_PREFIX = "校正后的人声："
SUMMARY_DISCLAIMER = "计划覆盖范围与实测波形质量是两件事，二者都不代表通过了听感验收。"

# ── Status banner ─────────────────────────────────────────────────
STATUS_COMPLETED = ("已完成", "所有计划区段都由有支持的对应关系合成。")
STATUS_UNRESOLVED = ("已完成，但存在未解决区段",
                     "部分内容被静音、模糊或未匹配。信任结果前请先检查对应关系与警告。")
STATUS_ANALYSIS_ONLY = ("分析完成", "只输出了检视用产物，没有合成音频。")
STATUS_FAILED = ("失败", "管线在生成校正人声之前就中止了。")
STATUS_CANCELLED = ("已取消", "任务被停止；已完成的分析缓存仍可复用。")
STATUS_RUNNING = ("运行中", "任务尚未结束。")
STATUS_RUNNING_NOTE = "界面显示的是进行中的状态；已有产物可能来自更早的一次运行。"
STATUS_UNKNOWN = "未知"
# A directory may hold artifacts without a job manifest: a partial run, or files
# copied out of one. The summary says so instead of denying what it can read.
STATUS_NO_MANIFEST_WITH_ARTIFACTS = "无任务清单，但存在产物"
STATUS_NO_MANIFEST_NOTE = ("该目录没有 job.json，因此无法给出这次任务的状态；"
                          "下面的指标、对应关系与警告来自目录中实际存在的文件。")
STATUS_UNKNOWN_NOTE = "状态异常，请检查 job.json。"

# Headlines for the runner's own finish status, which is about the process rather
# than the artifact: a job can "succeed" and still leave regions unresolved.
RUN_STATUS_HEADLINES = {
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}

# ── Metric row labels ─────────────────────────────────────────────
ROW_LYRIC_UNITS = "歌词单元"
ROW_LYRIC_UNITS_VALUE = "教师 {teacher} · 学生 {student}"
ROW_UNIT_STATUS = "单元状态"
ROW_PLANNED_COVERAGE = "计划歌词覆盖率"
ROW_SEMANTIC_MATCH = "精确语义匹配率"
ROW_UNSUPPORTED_REGIONS = "未支持区段"
ROW_AMBIGUOUS_MATCHES = "模糊匹配"
ROW_TEACHER_VOICED_COVERAGE = "教师有声语义覆盖率"
ROW_EXPORTED_LENGTH = "导出长度"
ROW_LENGTH_EXACT = "精确"
ROW_LENGTH_MISMATCH = "不一致"
ROW_LENGTH_VALUE = "{count} 采样（{state}）"

PITCH_VS_TEACHER = "对比教师"
PITCH_VS_TEACHER_REGISTER = "对比教师（换算到学生音区）"
PITCH_VS_TARGET = "对比渲染目标"
PITCH_ROW_ERROR = "音高误差（{label}）"
PITCH_ROW_RECALL = "有声召回率（{label}）"
PITCH_ROW_VALUE = "中位 {median}，p95 {p95}，基于 {frames} 个共同有声帧"

TIMING_ROW_COMPARED = "可靠比对的单元"
TIMING_ROW_COVERAGE = "全部教师单元中落在容差内"
TIMING_ROW_ONSET = "起始误差"
TIMING_ROW_OFFSET = "结束误差"
TIMING_ROW_COMPARED_VALUE = "{reliable} / {total}"
TIMING_ROW_ERROR_VALUE = "均值 {mean} · p95 {p95}"

ROW_SYNTHESIS_DEVICE = "合成设备"
ROW_RENDERED_RATIO = "已渲染样本比例"
ROW_VOICED_RATIO = "有声帧比例"
ROW_TIMING_CHANGED = "是否改变节奏"
ROW_PITCH_POLICY = "音高策略"
ROW_REGISTER_OFFSET = "音区差（按持续音测量）"
ROW_NOTE_SHIFTS = "逐词移动的词数"
ROW_NOTE_SHIFTS_VALUE = "{moved} / {measured} 个"
ROW_MUTED_REGIONS = "静音区段"
ROW_ACOUSTIC_TRANSITIONS = "声学过渡"
ROW_PEAK_MEMORY = "峰值显存"
ROW_JOB_TIME = "任务耗时"
VALUE_YES = "是"
VALUE_NO = "否"
VALUE_SEMITONES = "{value:+.2f} 半音"
VALUE_SECONDS = "{value:.3f} 秒"

WARNING_LIMIT_PREFIX = "测量限制："
WARNING_STRUCTURAL_FAILURE = "导出波形未通过结构校验。"

# ── Pitch policies, region statuses and modes ─────────────────────
PITCH_POLICY_LABELS = {
    "none": "不改变音高（学生原样）",
    "student_contour_student_register": "学生曲线 · 逐词对到教师音级 · 学生音区（适应模式）",
    "teacher_contour_student_register": "教师曲线 · 学生音区",
    "student_contour_teacher_register": "学生曲线 · 教师音区",
    "teacher_contour_teacher_register": "教师曲线 · 教师音区（绝对音高）",
}

REGION_STATUS_LABELS = {
    "matched": "匹配",
    "homophone": "同音字",
    "substitution": "疑似替换",
    "missing": "缺失",
    "unresolved": "未解决",
    "unvoiced_gap": "无声音隙",
    "phonetic_only": "仅发音匹配",
    "extra": "多余",
    "ambiguous": "模糊",
}

REGION_MODE_LABELS = {
    "synthesize": "合成",
    "mute": "静音",
    "silence": "静音",
    "acoustic_fallback": "声学兜底",
}

REGION_REASON_LABELS = {
    "voiced_teacher_region_without_lyric_correspondence": "教师有声区段没有歌词对应",
    "teacher_lyrics_unavailable": "教师歌词不可用",
    "untranscribed_activity_no_lyric_correspondence": "未转写区段没有歌词对应",
    "pitch_unvoiced_not_acoustic_silence": "无音高不等于静音",
    "multiple_lexically_equivalent_paths_progress_is_only_weak_evidence": "存在多条等价路径，仅凭进度无法判定",
    "consonant_limits_infeasible": "辅音时长限制下无法排布",
    "no_phone_timeline_in_window": "该窗口没有音素时间轴",
    "no_register_offset": "没有可用音区差",
    "already_at_target_level": "已在目标音级",
    "no_teacher_voiced_level": "教师该区段无有声音高",
    "no_student_voiced_level": "学生该区段无有声音高",
    "teacher_contour_used": "使用教师曲线，不逐词位移",
    # The pipeline reuses a few status names in the reason field, plus these
    # explanation-only reasons that appear alongside them.
    "matched": "正常匹配",
    "homophone": "同音字",
    "possible_lyric_mismatch_or_asr_error": "可能歌词不一致或识别有误",
    "no_corresponding_student_lyric_unit": "没有对应的学生歌词单元",
}

# ── Audio clip labels ─────────────────────────────────────────────
AUDIO_TEACHER = "教师参考"
AUDIO_STUDENT = "学生原始"
AUDIO_CORRECTED = "校正后学生"
AUDIO_IDENTITY = "identity 重建"
AUDIO_PITCH_ONLY = "仅改音高"
AUDIO_TIME_ONLY = "仅改节奏"

# ── Stages and progress ───────────────────────────────────────────
STAGE_LABELS = (
    ("analysis", "分析：歌词、音素、音高与音符"),
    ("alignment", "对应：匹配歌词单元并建立时间映射"),
    ("render", "合成：变换特征并渲染校正后的人声"),
    ("verification", "校验：重新测量导出的波形"),
    ("timing_verification", "时间核对：重新识别输出词边界"),
    ("review", "汇总：整理音频、警告与来源信息"),
)
STAGE_TRACK_LABELS = {"teacher": "教师", "student": "学生"}
PHASE_LABELS = {"Lyrics": "歌词", "Phonemes": "音素", "Notes": "音符"}
DETAIL_CHUNK = "{phase}：{track} 第 {index}/{total} 块"
DETAIL_PITCH = "提取连续音高：{track}"
DETAIL_RENDER = "正在渲染：{mode}"
DETAIL_LOADING_VOCODER = "正在加载声码器（{device}）"
DETAIL_STARTING = "正在启动管线。"
DETAIL_RUNNING = "运行中…管线输出显示在日志页。"
DETAIL_FINISHED = "已结束：{status}，耗时 {seconds:.1f} 秒。"
DETAIL_CANCELLED = "已取消。已完成的分析缓存仍可复用。"
DETAIL_CANCELLING = "正在取消，等待各工作进程停止。"

# ── Status bar and dialogs ────────────────────────────────────────
STATUSBAR_READY = "就绪。"
STATUSBAR_RUNNING = "运行中。"
STATUSBAR_JOB = "任务{status}。"
STATUSBAR_PLAYBACK_UNAVAILABLE = "无法播放：{message}"
STATUSBAR_PREFS_FAILED = "无法保存界面偏好设置。"
DIALOG_CANNOT_START = "无法开始"
DIALOG_PIPELINE_FAILED = "管线失败"
DIALOG_JOB_RUNNING = "有任务正在运行"
DIALOG_REUSE_TITLE = "复用已有分析"
DIALOG_REUSE_BODY = ("复用已有分析不会因为分析设置或算法变化而重跑分析，合成阶段仍会重新读取输入音频。\n\n"
                     "是否继续？")
DIALOG_JOB_RUNNING_BODY = "停止正在运行的任务并关闭窗口？"
DIALOG_FAILED_FALLBACK = "管线在合成之前就中止了。"
DIALOG_FAILED_BODY = ("{error}\n\n日志页与输出目录中的 processing.log 记录了管线的输出。")
BULLET_ITEM = "• {item}"

# ── Job validation messages ───────────────────────────────────────
VALIDATE_CHOOSE_RECORDING = "请选择{label}录音。"
VALIDATE_FILE_NOT_FOUND = "{label}文件不存在：{path}"
VALIDATE_SAME_FILE = "教师录音与学生录音不能是同一个文件。"
VALIDATE_CHOOSE_OUTPUT = "请选择输出目录。"
VALIDATE_OUTPUT_IS_FILE = "输出路径已存在且是文件：{path}"
VALIDATE_CONFLICT = "「只做分析」与「复用已有分析」不能同时启用，请取消其中一个。"
VALIDATE_MISSING_ANALYSIS = "复用已有分析需要输出目录中存在：{files}。"
VALIDATE_RANGE = "{name}必须大于 {low} 且不超过 {high}。"
LABEL_TEACHER_SHORT = "教师"
LABEL_STUDENT_SHORT = "学生"

# ── Pipeline sentence translations ────────────────────────────────
# The workers write these in English; the interface shows them in Chinese. An
# unmatched sentence is returned as-is, never dropped.
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
    # Timeline statements: these appear when the output keeps the student's clock.
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
            values = [STAGE_TRACK_LABELS.get(value, value) for value in match.groups()]
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

# Sentences that carry values are matched by shape and rebuilt in Chinese. Each
# pattern is anchored so it cannot swallow an unrelated sentence.
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
    """Drop the repr quotes the workers put around a word, then use Chinese ones.

    The pipeline writes the word with repr(), so the English sentence already
    carries ASCII quotes. Re-quoting that verbatim produced doubled marks, so the
    outer pair is removed and re-applied as full-width quotes.
    """
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return f"“{text}”"


def warning_text(sentence: str) -> str:
    """Translate one pipeline sentence, or return it unchanged.

    Unrecognized text stays in English on purpose: a warning the interface has not
    been taught about is exactly the one a user must still be able to read.
    """
    text = str(sentence)
    if text in EXACT_WARNINGS:
        return EXACT_WARNINGS[text]
    if text in EXACT_LIMITATIONS:
        return EXACT_LIMITATIONS[text]
    for pattern, template, groups in TEMPLATE_WARNINGS:
        match = pattern.match(text)
        if match:
            values = list(match.groups())
            if groups == ("word", "ratio"):
                # The English message names the word first, then the ratio.
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


def label(kind: str, table: dict) -> str:
    """Look up a Chinese label, falling back to the raw value."""
    return table.get(kind, kind)


def status_label(kind: str) -> str:
    return label(str(kind), REGION_STATUS_LABELS)


def mode_label(kind: str) -> str:
    return label(str(kind), REGION_MODE_LABELS)


def reason_label(kind: str) -> str:
    return label(str(kind), REGION_REASON_LABELS)


def policy_label(policy: str) -> str:
    return label(str(policy), PITCH_POLICY_LABELS)
