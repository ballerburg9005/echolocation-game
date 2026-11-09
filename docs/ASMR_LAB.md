# ASMR experiment — physically rasterized SVG heads in real 2D FDTD

**Separate branch:** `experiment/asmr-fdtd-wav-path-lab`  
**No changes to `main` or the original arcade mode.**

## Build and run

The input recording is always **`~/test.mp3`**. No command-line audio file
parameter is needed; the system `libmpg123` decoder feeds the moving
**FDTD mouth emitter**, not the speakers directly.

```bash
git fetch origin
git switch experiment/asmr-fdtd-wav-path-lab
git pull --ff-only
sudo apt-get install -y libmpg123-dev libsdl2-dev librsvg2-dev libcairo2-dev pkg-config
make test
make asmr
./build/asmr_room
```

To supply or replace your private input: copy it to `~/test.mp3` and press
**L** to reload. The recording need not be committed to GitHub.

Both head drawings have physically specified **millimetre** viewBoxes and
are rendered with librsvg/Cairo at sub-millimetre spatial detail, including
your original bezier curves, elliptic arcs and SVG transforms. The program
loads the user's copy from `~/` when present, otherwise the copies
checked into `asmr_shapes/`.

### Testing silence or missing pressure waves

Press **D** to toggle a built-in one-second repeating 320–2720 Hz
calibration chirp. It goes into the *same moving-mouth CUDA FDTD
pressure solver* as `~/test.mp3`, not directly to an audio output.
The chirp is available even when the MP3 is absent and is only enabled
when explicitly requested; **D** again returns to normal MP3 input.
Look for `[ASMR FDTD]` diagnostics every five seconds: `MP3=`,
`preview=`, `probe=`, `earPeak=` and queue depth. If `earPeak`
is nonzero but headphones remain silent, check SDL's selected output
device, system volume and mute. If `earPeak` stays zero on the
calibration tone, that points to a source/mask/FDTD problem instead.
The visible cyan/magenta field in V mode is the same solver field.

### Head geometries and eardrums

**The default listener is YOUR `listener_v2.svg`** — both mirrored halves
and the large pinna openings, rather than the old simple round head.
Its original **185.348 × 136.859 mm** viewBox maps to exactly
**0.185348 × 0.136859 m** in the FDTD field, centred at (1m,1m).
The two authored half-paths are stitched only across their central
straight seam. Their contours are static sound-reflecting SOLID cells.

The **default V2** probes have been moved **outward into the real
open pinna recesses**, at **(0.938 m, 1.003 m)** and **(1.062 m,
1.003 m)**, **12.4 cm apart**. The prior ±4 cm probes were literally
inside the V2 SVG solid; the program cut artificial ear passages to
reach them. V2 no longer cuts those passages: the probes are in
SVG-authored air, and the GPU sampler requires virtually its entire
bilinear stencil to remain in air. This is an **outer-recess microphone
experiment**, not the same as an 8 cm eardrum pair.

To preserve your earlier **8 cm child eardrum-to-eardrum** assumption,
press **H** for `listener_with_ear_channel.svg`: that independent
model retains the 8 cm target. It is an experimental dimensional
assumption, not a universal claim about children's anatomy. At too
coarse a resolution, the program refuses a microphone position that
cannot sample air reliably.

Press **H** in live mode to cycle the alternative geometries:
`listener_v2.svg` (default), `listener_with_ear_channel.svg`, and
`listener.svg`. The previous simple listener uses the experimental
8 cm + 0.8×2 cm ear-canal allowance, giving **9.6 cm outer-recess mic
spacing**. Only the explicit ear-channel mode uses the 8 cm target.

The moving artist is YOUR **`speaker.svg`** (134.483 × 130.034 mm),
rotated so its recessed mouth always faces the listener's centre.
The mouth lies about **5.5 cm** ahead of the artist centre, physically
inside the open front recess. It emits MP3 pressure in the same solver.
The speaker silhouette additionally creates **genuine moving
fractional impedance scattering** sampled from the uploaded SVG
alpha texture in the CUDA stencil. It is *not* a moving hard mask:
pressure is preserved during motion to avoid numerical whips.
This is an approximation to a moving solid boundary, not a complete
fluid–structure or rigid-body acoustic calculation.

The mouse still maps 1:1 to the full room, subject to listener,
speaker and outside-wall collision clearance. A physical
**25 cm ruler** appears at the bottom of the view.

### Frequency and full-room physical scale

| Property | Default live | Offline 5/10/20 s render |
|---|---:|---:|
| Physical room | 2.0 × 2.0 m | 2.0 × 2.0 m |
| Grid cell | **3.5 mm** | **1.5 mm** |
| Acoustic cells per side | ~571 | ~1333 |
| Wave steps/s (CFL) | ~139k | ~323k |
| Ten-cells/wavelength reference | ~9.8 kHz | ~22.9 kHz |
| Source low-pass | **8.5 kHz** | **19 kHz** |
| Sound capture | two FDTD microphones | same two microphones |

Press **Q** to cycle 3.5 / 3 / 2.5 / 5 / 8 / 10 mm live grid
settings, with corresponding input passbands. Offline stays at 1.5 mm,
independently of the preview resolution; it may take substantially
longer than realtime to render even a short clip.

The room dimensions are **not 50 cm**. Every five seconds the terminal
prints `[METRES]` showing source position in metres, actual left/right
source-to-microphone distance, predicted flight time using
343 m/s, exact 2m room side, dx and dt. The 2D pressure field
decays differently from 3D spherical waves — that can make distance
changes feel weaker, even when the grid scale is correct. We have
not added artificial 3D distance gain or HRTF processing.

### Controls

| Control | Action |
|---|---|
| Mouse | Position the artist absolutely in the 2 × 2 m room |
| M | Toggle mouse versus WASD/arrow movement |
| V | Toggle the **actual CUDA pressure-wave visualization** |
| 9 / 0 | Decrease/increase wave exposure, not the physical pressure |
| H | Cycle the three *real FDTD* listener SVG shapes, default V2 |
| Q | Cycle real-time acoustic grid resolution |
| F5 / 5 | Render and then **loop exactly 5 s** |
| F6 / 1 | Render and then **loop exactly 10 s** |
| F7 / 2 | Render and then **loop exactly 20 s** |
| P | Restart the most recently completed replay loop |
| **Space** | Stop playback and return to live movement/recording |
| D | Toggle in-solver FDTD test chirp even without MP3 |
| T | Toggle live FDTD audio preview |
| L | Reload `~/test.mp3` |
| R | Reset live wave field and path history |
| Esc | Quit |

### Exact-duration replay and visualization

The input path is recorded in wall-clock seconds, with
timestamps and artist positions. Pressing a render button selects
the last 5/10/20 seconds available, then runs the offline **same
pressure-stencil FDTD** in fixed acoustic time steps. The stereo
resampler's output is **trimmed or padded to exactly N × 48,000
interleaved stereo frames**, where N is 5, 10 or 20. This corrects
the old discrepancy where a graph-batch overrun or acoustic output
tail could make the requested length unclear. If the path has less
history than requested, the artist holds its last captured position
for the remainder.

The complete rendered stereo buffer is saved as
`asmr_last_05s.wav` / `asmr_last_10s.wav` /
`asmr_last_20s.wav`, with the companion timestamped
`asmr_last_XXs_path.csv`. These are **outputs only**: MP3 remains
the default input.

Playback begins only after the full offline calculation finishes.
The next loop copy is queued into SDL **before** the current copy
runs out; audio playback does not wait for a CUDA kernel.

During the offline render, 24 frames/second of the **actual CUDA
pressure field** are also saved at a reduced 256×256 *display-only*
resolution. During looping replay, SDL's consumed audio-frame count
advances these snapshots in sync with the animated speaker. This
eliminates the old frozen final-pressure image in playback without
changing the 1.5mm FDTD simulation, 48kHz stereo buffer, or MP3 input.
The snapshots are a visual record; no sound is synthesized from them. The
visible artist SVG follows the recorded path again, the bottom
playhead counts each exact loop, and the title shows elapsed/
total seconds. Press **Space** to leave looping playback.

The cyan/magenta waves are the exact CUDA pressure field read from
the GPU, not a visualization-only oscillator. Solid listener head
walls are drawn as hard reflecting geometry. The 2D shader displays
raw signed pressure with adjustable exposure.

## Acoustic limitations and validation

- This is **genuine two-dimensional FDTD**, not HRTF rendering,
  generic reverb, direct stereo panning or ray tracing. The two
  microphones sample the propagated field on opposite sides of the
  SVG head. CPU regressions separately test two-ear arrival order.
- The V2 probes are in its existing pinna recesses, not synthetic
  V2 ear canals. The alternative ear-channel model remains a 2D
  acoustic approximation, not a measured anatomical canal. Pinna-shaped
  2D outlines do not reproduce real vertical front/back HRTFs.
- Moving-speaker scattering uses fractional soft impedance instead
  of an energy-conserving moving boundary; no hard pressure snaps.
  This preserves source motion without pretending to solve 3D
  fluid–structure coupling.
- Outer walls use a lower broad-band reflection coefficient for
  rubber-like attenuation; no measured rubber frequency response.
- The GPU load must be benchmarked in actual runtime, not inferred
  from its board's wattage. Larger 1.5mm offline grids require much
  more work even though completed audio plays at normal speed.
- Use low headphone gain during initial tests. Numerical code
  compilation and offline CPU geometry checks cannot verify physical
  loudness or your subjective ASMR response.

The ASMR hypothesis paper motivates exploring proximity and sound
movement but **does not establish** that any particular motion will
induce ASMR:

https://doi.org/10.3389/fnbeh.2025.1688172
