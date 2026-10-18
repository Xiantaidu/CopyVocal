# Local resources and reproducibility

CopyVocal runs with its physically bundled Windows CPython 3.12 interpreter:

```powershell
.\python\python.exe scripts\audit_local_resources.py
```

The audit reads the local files and writes `settings/local_resource_audit.json`
and `settings/requirements-lock.txt`. It does not copy upstream resources or
load inference models. Do not rerun `bootstrap.py` over an edited checkout:
that script is an initial migration tool and can overwrite adapted code.

The completed audit on 2026-10-15 checked 32,744 files in 2,450 directories,
covering 14,605,980,576 bytes. It found no symbolic links, directory junctions,
reparse points, or files with multiple hard links. SHA-256 hashes were recorded
for 30,703 files; 2,041 regenerable Python bytecode files were checked for links
and excluded from hashing. All 72 previously recorded model/native-resource
hashes matched the original physical-copy manifest. The audit JSON contains
the full file inventory, hashes, import origins, native DLL paths, and results.

That inventory predates the Japanese model migration. The subsequent dedicated
audit in `settings/japanese_resource_audit.json` verified 27 Japanese/shared
files against the upstream bytes, including five newly copied files totaling
306,397,497 bytes. Every new model file has one link and no reparse point.
`settings/resource_manifest.json` now records 77 model/native-resource files.
The reproducible migration audit is:

```powershell
.\python\python.exe scripts\audit_japanese_resources.py --source D:\Vocal2Midi
```

This command uses the source only for migration comparison; inference resolves
the project-local copies. Existing pyopenjtalk native modules and its Open JTalk
dictionary are local as well. A CPU model-loading/G2P resource check succeeded
before the user requested that audio testing be deferred; no Japanese audio
correction or singing-quality result is claimed.

## Bundled assets

| Asset | Local directory | Files | Bytes |
| --- | --- | ---: | ---: |
| Qwen3-ASR encoder and GGUF decoder | `models/Qwen3-ASR-1.7B-dml` | 3 | 4,708,336,117 |
| TiFA and its language assets | `models/tifa-1.0-onnx` | 17 | 178,478,959 |
| Japanese RomajiASR | `models/romajiASR` | 3 | 189,338,441 |
| Japanese kashi G2P and lexicon | `models/kashi-g2p-onnx` | 2 | 117,059,056 |
| GAME | `models/GAME-1.0.3-medium-onnx` | 6 | 195,977,610 |
| RMVPE | `models/RMVPE` | 1 | 361,688,443 |
| PC-NSF-HiFiGAN | `models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` | 5 | 56,673,770 |
| Portable Python, packages, and native libraries | `python` | 32,671 | 8,986,161,004 |
| Qwen native inference binaries | `inference/qwen3asr_dml/bin` | 40 | 116,342,753 |
| FFmpeg 7.1.1 | `ffmpeg.exe` | 1 | 2,321,920 |

Copy provenance is retained in `settings/resource_manifest.json` and
`settings/synthesis_runtime_manifest.json`. Those files name the original
source locations, including the former upstream project and model drive.
These strings record where copying happened; they are not runtime search paths.

## Runtime isolation

`python/python312._pth` explicitly lists only paths under this project and does
not execute `site.main()`. The audited process has `isolated=True`,
`ignore_environment=True`, and `no_site=True`. Consequently the user
`site-packages`, machine Python environment, and `PYTHONPATH` are excluded.
All seven `sys.path` entries and all loaded Python module origins were local.
The audit imported 14 critical dependency modules, including NumPy, SciPy,
librosa, SoundFile, ONNX Runtime, tokenizers, and PyTorch. No import came from
an upstream checkout, another Python environment, or the user profile.

The process loaded 189 project-native libraries and 58 Windows system/driver
libraries. No other native-library origin was observed. Windows and compatible
hardware drivers remain operating-system prerequisites. The PyTorch runtime is
`2.8.0+cu129`; CUDA models and DirectML sessions are validated by separate
end-to-end processing runs rather than initialized by this audit.

Runtime-source scanning found no executable-source literal referring to the
upstream project drive or model drive. Initial bootstrap tools and provenance
records are intentionally excluded. The audit does not rename or disconnect
the user's original directories, and its report does not claim that destructive
test was performed.

## Environment snapshot

`settings/requirements-lock.txt` records all 71 installed distributions from
the actual portable runtime using `pip freeze --all`. This includes inherited
optional packages and UI libraries; it is not a minimal dependency list or an
offline wheel archive. Versions alone cannot recreate native model binaries,
model weights, dictionaries, or a CUDA build from the standard PyPI index.
Preserve the resource directories and their manifests for exact offline reuse.
The full SHA-256 inventory provides the artifact identity that the version list
cannot provide.

Only one ONNX Runtime distribution should be installed in this runtime:
`onnxruntime-directml==1.24.4`. Installing another ONNX Runtime variant into the
same directory can replace shared files. The bundled PyTorch package carries
its CUDA runtime libraries and does not require a machine CUDA toolkit.

## Licenses and retained notices

License scope is recorded per component. A repository's software license does
not automatically grant the same license to separately distributed weights,
datasets, language dictionaries, or bundled binaries.

| Component | Retained local evidence | Scope and status |
| --- | --- | --- |
| Adapted Vocal2Midi source | `licenses/Vocal2Midi-LICENSE`, `licenses/Vocal2Midi-NOTICE`, `licenses/Vocal2Midi-ACKNOWLEDGEMENTS.md` | Apache-2.0 for repository software; retained third-party terms also apply. |
| Adapted PC-NSF generator source | `inference/vocoder/LICENSE` and source attribution | Apache-2.0 source adaptation from hifisampler/SingingVocoders; modified files identify changes. |
| PC-NSF model weights | Model-local `NOTICE.txt`, `NOTICE.zh-CN.txt`, and `STATEMENTS.txt` | OpenVPI explicitly identifies CC BY-NC-SA 4.0 for these weights. All supplied statements were physically preserved. |
| Qwen3-ASR model bundle | The three copied model files; Vocal2Midi integration attribution | No model-specific license file was included in the supplied bundle. Verify the exact converted artifacts against [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) and their distributor before redistribution. |
| GAME model bundle | Config and five ONNX files; Vocal2Midi attribution | The supplied bundle lacks a model-specific license file. Consult [GAME](https://github.com/openvpi/GAME) for the exact weight release terms. |
| TiFA model and language assets | Local config, ONNX models, lexicons; Vocal2Midi attribution | The supplied model directory lacks a model-specific license file. Consult [TiFA](https://github.com/openvpi/TIFA). Dictionary-specific notices remain in the adapted inference tree. |
| RMVPE ONNX weights | The copied ONNX file and integration attribution | The supplied bundle lacks a weight-specific license notice. Consult [RMVPE](https://github.com/Dream-High/RMVPE) and the ONNX converter/distributor; source-code terms alone do not identify the converted weight license. |
| Japanese RomajiASR and kashi G2P | Original integration attribution in `licenses/Vocal2Midi-NOTICE` and `licenses/Vocal2Midi-ACKNOWLEDGEMENTS.md`; copy hashes in the resource manifest | These supplied model bundles contain no model-specific LICENSE/NOTICE file. No weight license is inferred from the integration code. |
| PyTorch | `python/Lib/site-packages/torch-2.8.0+cu129.dist-info/LICENSE` and `NOTICE` | Package and bundled third-party notices were preserved by physical copying. |
| Python | `python/LICENSE.txt` | CPython and embedded notices remain local. |
| Other Python packages | Their installed `*.dist-info` license/metadata directories | These are included in the file/hash audit; the requirements snapshot identifies exact installed versions. |
| Mandarin/Cantonese dictionaries | `inference/TiFA/g2p/converters/cpp_pinyin/dicts/*/License.txt` and `inference/LyricFA/Dicts/*/License.txt` | Original dictionary notices are retained. |
| FFmpeg | `licenses/Vocal2Midi-NOTICE` and recorded `ffmpeg -version` build configuration | The supplied build has no `--enable-gpl` or `--enable-nonfree` switch. Repository attribution is retained; this audit does not certify a complete binary redistribution/source-offer package. |

Attempts to retrieve missing upstream license texts during this audit failed at
TLS connection setup. No missing license text was invented or silently
substituted. The local technical isolation audit passes; license completeness
for the four model bundles without weight-specific notices remains unverified.
