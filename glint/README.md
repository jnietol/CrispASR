# glint (in-tree copy)

Our clean-room MP3 (MPEG-1/2/2.5 Layer III) + AAC-LC (ADTS,
experimental phase 1) encoder in C++17, MIT licensed. This is an
in-tree copy of the sibling `glint` repository — encoder core only
(`src/` + `include/glint/glint.h`); the upstream CLI, unit tests,
language bindings and packaging live in the glint repo and are not
carried here. Same relationship as `ggml/`: develop in the source
repo, sync the core here.

Used for TTS/S2S compressed output:

- `crispasr-server` — `POST /v1/audio/speech` with
  `response_format=mp3` or `response_format=aac`
- `crispasr-cli` — `--tts-output out.mp3` / `out.aac`
  (and `--s2s-output`)

Both go through `examples/cli/crispasr_mp3_writer.h` /
`crispasr_aac_writer.h`, which prepend the ID3v2 AI-provenance tag
(`crispasr_make_id3v2_ai_tag`) to the encoded stream.

MP3/AAC output is always available, no system package required. When
the build finds libmp3lame it stays in as an optional MP3 fallback:
used automatically if glint fails, or forced with
`CRISPASR_MP3_ENCODER=lame` for A/B comparisons.

Syncing is automated: the `sync-glint` GitHub workflow pulls the
committed state of `CrispStrobe/glint` main (on repository_dispatch
from glint pushes, daily schedule, or manual dispatch), regenerates
the CMake source list and the sync marker below, runs the
`test_tts_provenance "[mp3],[aac]"` suite, and pushes to main. The
`validate-glint-fresh` release job fails a release whose in-tree copy
is behind glint main. To sync by hand, run `tools/sync-glint.sh` —
never `cp` from a glint working tree (it may hold another session's
WIP; the script always takes a committed state).

Synced at upstream commit: `2576102767ab1fd871a3642cb670df7f24ed3b67` (docs: rewrite README — current, ordered, stale history removed).
