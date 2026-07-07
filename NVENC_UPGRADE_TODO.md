# GPU encoding status for the triggered video saver

**Tracking issue: https://github.com/strawlab/strand-braid/issues/29** (filed 2026-07-02, still open)

## Status as of 2026-07-06 (strand-braid 1.0.0-rc.5): RESOLVED IN PRACTICE

Full-rate color GPU encoding **works** via strand-cam's `Ffmpeg` codec piping to the
system ffmpeg's `h264_nvenc`. Verified on this rig: 105 fps sustained for 60+ s at
1920x1200 color, zero dropped frames, 1 s pre-trigger buffer intact, correct colors.
`video_config/triggered_video_saver.toml` now ships this configuration:

```toml
mp4_codec = { Ffmpeg = { codec = "h264_nvenc" } }
mp4_max_framerate = "Unlimited"
```

What changed: rc.5 fixed two of the three bugs from issue #29 — mp4-writer errors no
longer crash strand-cam, and the colorspace problem that blocked Bayer (color) frames
from reaching ffmpeg was fixed. ffmpeg 6.1's own nvenc (SDK-12-era headers) always
worked on this Blackwell GPU; only strand-cam's built-in bindings are too old.

## Still pending upstream (the original issue)

The **built-in** `mp4_codec = "H264Nvenc"` still fails on Blackwell GPUs
(`is_nvenc_functioning: false` — bindings still built against Video Codec SDK 11.0.10;
Blackwell needs 13.0+). This no longer blocks anything here, but when a future
strand-braid release updates the bindings, the built-in codec would remove the ffmpeg
dependency. Healthcheck after any strand-braid upgrade:

```bash
strand-cam --no-browser --camera-backend pylon --camera-name Basler-41942900 \
  --http-server-addr 127.0.0.1:3441 --trusted-network 127.0.0.1/32 &
sleep 8
curl -sN -H 'Accept: text/event-stream' http://127.0.0.1:3441/strand-cam-events \
  | grep -m1 -o '"is_nvenc_functioning":[a-z]*'
kill %1
```

If it ever reports `true`, `mp4_codec = "H264Nvenc"` becomes an alternative to the
Ffmpeg route (equivalent output; one less moving part).

## Fallbacks (if ffmpeg/nvenc is unavailable on some machine)

- `mp4_codec = { Ffmpeg = { codec = "libx264", post_codec_args = [["-preset","ultrafast"],["-crf","21"]] } }` (CPU)
- `mp4_codec = "H264OpenH264"` with `mp4_max_framerate = "Fps30"` (built-in software
  encoder ceiling ~37 fps at 1920x1200)
