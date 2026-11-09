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
The startup console prints `[AUDIO CLOCK]` with the actual SDL output
format; it must be 48,000 Hz, stereo float. Every five seconds,
`generated/wall` estimates how many seconds of live audio were
produced per wall-clock second, and `solver/compute` gives an upper
bound on solver throughput, including GPU transfer/synchronization
but not UI rendering. A `solver/compute` figure below 1x suggests
hardware cannot keep pace; a nearly empty queue plus frequent
`near-empty queues` supports underruns. Neither causes the device
sample rate to change automatically. A low-pass at 6kHz in the economical live preset, 2D room
resonances and partially transmissive artist geometry may instead
color the live timbre. The **offline 1.5mm/19kHz WAV is fully
precomputed** and therefore does not play at a lower pitch when
rendering takes longer than real time.

### Portrait room and live crackle investigation

The room is **2.0m wide × 3.5556m long**, an actual 9:16
**rectangular FDTD grid** rather than a stretched picture. The
listener is exactly at **(1.000m, 1.185m)** — one-third up from the
bottom — leaving **2.370m** in front (+Y) and **1.185m** behind,
with full width to circle either side. The real mouth-to-ear range
near the far wall is shorter than 2.370m because the speaker must stay
inside the wall and its mouth points 5.5cm toward the listener.
Mouse movement, collisions, 25cm ruler, source-to-mic travel times,
SVG physical dimensions, and offline paths share these metre units.
The window is **540 × 960**. Signed-pressure visual readback runs at
up to **30 fps** with a **360 × 640** preview, whereas the solver still
runs at its required physical rate. The completed offline WAV keeps
**1.5mm FDTD grid and 19kHz passband**, but with more cells it may
need longer to render; 9:16 replay snapshots are 180 × 320 pixels.

The default live grid is **5mm**, approximately **400 × 711 = 284,400**
cells, with a 6kHz input filter. The previous 3.75mm square-room
preview was **533 × 533 = 284,089** cells. The total grid budget has
therefore been **preserved to within 0.11%**, not multiplied by the
9:16 transformation; the coarser dx also reduces time steps per
simulated second. This yields about **25% fewer live cell updates per
simulated second** than that old square-room preview even though the
physical area is larger. This estimate is not a
promise about GPU wattage or actual frame rate. Press **Q** to select
4.5mm, 3.75mm, finer or coarser settings if useful. The audio queue
aims to retain **120ms** of stereo audio to tolerate scheduling jitter.
Audio continues at 48kHz regardless of FDTD cell size. The lower
live low-pass can sound softer; always compare it to a full-quality
**F5 offline replay** before blaming the acoustic boundary shape.

The crackle is **not yet proven** to be GPU starvation. Compare
`solver/compute` (solver capacity), `near-empty queues` (starvation),
`safety-clipped samples` (output clipping), `earPeak` and the `D`
calibration chirp, then compare live sound to the finished WAV. If
crackling persists offline at normal headphone level, investigate the
moving emitter, sample reconstruction and geometry rather than just
increasing the live buffer or GPU power.

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
That means a visible, attenuated pressure field INSIDE the moving
speaker, whereas the static listener's hard mask has zero pressure
inside. This is expected from the different boundary conditions,
not evidence of hidden sound processing. We deliberately have not
made the moving speaker rigid just to hide those waves: that could
reintroduce discontinuities during movement. This is an approximation
to a moving solid boundary, not a complete
fluid–structure or rigid-body acoustic calculation.

The mouse still maps 1:1 to the **target location** in the full
physical room, subject to listener, speaker and outside-wall collision
clearance. For the same impedance geometry and mouth injection to
appear where the CUDA solver actually computes them, the **orange
speaker SVG now tracks the smoothed acoustic-source position**.
When the mouse moves faster than the source speed limit, a **small
hollow yellow ring** shows the desired mouse position. This replaces
the old misleading ghost where the speaker sprite jumped far ahead
of the actual acoustic speaker head, sometimes by hundreds of pixels.
The underlying physical source smoothing and 1:1 mouse target mapping
are unchanged. During looping offline replay, the head also follows
its **recorded physical FDTD movement**, not the unsmoothed cursor path.

When **V** is enabled, the displayed listener is the **actual pixelated
acoustic mask only**. Rendering the authored SVG over that same mask
made it look like there were two offset heads. Press **V** again to
see the original high-resolution SVG alone. CPU tests verify the SVG
solid occupancy and the pixelated mask agree in the shared metre-scale
coordinates and that the 9:16 texture lookup resolves the listener
position within two grid cells.

A physical **25 cm ruler** appears at the bottom of the view. The
window title includes the computed **mouth-to-listener range (metres)**
and how far the physics source is behind its mouse target. The terminal
`[METRES]` lines show the listener, mouth, microphone ranges and
predicted propagation delay. This allows auditing the actual geometry
without judging range solely by how loud the 2D acoustics seem.

### Frequency and full-room physical scale

| Property | Default live | Offline 5/10/20 s render |
|---|---:|---:|
| Physical room | **2.0 × 3.56 m** | **2.0 × 3.56 m** |
| Grid cell | **5.0 mm** | **1.5 mm** |
| Acoustic grid X × Y | **400 × 711** | **1333 × 2370** |
| Wave steps/s (CFL) | ~97k | ~323k |
| Ten-cells/wavelength reference | ~6.86 kHz | ~22.9 kHz |
| Source low-pass | **6.0 kHz** | **19 kHz** |
| Sound capture | two FDTD microphones | same two microphones |

Press **Q** to cycle 5 / 4.5 / 3.75 / 3 / 2.5 / 6 / 8 / 10mm live
resolutions (unsupported coarse-grid mic placements are skipped).
The default aims to free GPU time rather than simulate a physically
smaller room. The previous 3.75mm / 8.5kHz preview is still available
as a more demanding option. The high-quality offline grid remains
**1.5mm / 19kHz**, independent of the live resolution.

Every five seconds the terminal prints `[METRES]` with the actual
listener, mouse target, smoothed acoustic head, mouth, left/right
source-to-microphone distances and predicted flight times at **343m/s**.
The grid's physical width/height remain 2.0m × 3.5556m. **A true 2D
wave spreads as approximately 1/sqrt(distance)**, weaker attenuation
than the approximately 1/distance of a freely spreading 3D wave.
The ear outline also lacks full 3D pinna cues, so a 2m geometry may
sound subjectively closer even when the physical grid and acoustic
delays are correct. We have NOT added arbitrary distance gain to the
solver or audio; the numerical room scale is still auditable.

### Controls

| Control | Action |
|---|---|
| Mouse | Position the movement TARGET in 2 × 3.56 m; orange acoustic head follows smoothly |
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
pressure field** are also saved at a reduced 180×320 *display-only*
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
