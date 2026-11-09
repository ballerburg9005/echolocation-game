# ASMR experiment — physically rasterized SVG heads in real 2D FDTD

**Separate branch:** `experiment/asmr-fdtd-wav-path-lab`  
**No changes to `main` or the original arcade mode.**

## Build and run

The input recording defaults to **`~/test.mp3`**, but **`--mp3 FILE`**
selects any readable MP3 file (relative or absolute path). The system
`libmpg123` decoder feeds the moving **FDTD mouth emitter**, not the
speakers directly. **`--help`** documents all command-line options and
keys without starting a GUI, SDL or CUDA.

```bash
git fetch origin
git switch experiment/asmr-fdtd-wav-path-lab
git pull --ff-only
sudo apt-get install -y libmpg123-dev libsdl2-dev librsvg2-dev libcairo2-dev pkg-config
make test
make asmr
./build/asmr_room
./build/asmr_room --help
./build/asmr_room --mp3 "/path/to/recording.mp3"
```

Press **L** to reload whichever input you selected using `--mp3` or
the default `~/test.mp3`. The recording need not be committed to GitHub.

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
color the live timbre. The **offline 2mm/16kHz WAV is fully
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
**2mm FDTD grid and 16kHz input passband**, so its
**1000 × 1778 = 1,778,000 cells** also match the former square
**1333 × 1333 = 1,776,889** offline budget (within 0.1%).
The 2mm grid has a 17.15kHz ten-cells/wavelength numerical reference;
keeping 19kHz at that dx would compromise frequency accuracy. Thus
retaining a fixed cell count requires a modest trade-off in high-end
bandwidth. The finished 48kHz WAV remains fully precomputed and
crackle-free without live CUDA deadlines, while the 9:16 replay
pressure snapshots remain 180 × 320 pixels.

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

### True acoustic zoom and reflective room walls

Press **Z** to cycle **1.00× → 1.08× → 1.16× → 1.30× → SQUARE → 1.00×**.
These are gentle *actual FDTD* resolutions, not a 2×/4×/8×
timestep explosion. The original **2.0 × 3.5556m** full-size portrait
room is always available at 1.00×. The intermediate stages preserve
the **400×711 = 284,400** cells while refining dx 8%, 16% and 30%.
The final stage is a **1.8 × 1.8m physically square room** centered
on the listener, with approximately **533×533 cells**, essentially
the same GPU cell budget. The square room is drawn in a true square
540×540 viewport in the center of the 540×960 window; it is not
stretched to portrait. All stages preserve real sound speed, mic
locations and the unmodified SVG dimensions. The square stage has
about **1.48× the acoustic time steps per second** relative to 1.00×,
not 8×. Offline WAV rendering matches the selected physical room
at its independent high-quality cell budget. High zoom may still
cost too much for live audio; completed offline playback is unaffected.

Pressing **Z** at the last stage returns to the original giant room.
If the narrow channel mics cannot connect at 1.00×, Z explicitly
changes the displayed head back to V2 and logs that fact instead of
silently showing stale material. The window title states the active
head and zoom.

The outer walls now reflect sound with an approximate **R=0.82**
impedance rather than the previous R=0.32. Three asymmetrically placed
teal openings—one on the left wall, one on the right and one on the
front wall—use **R≈0.035 locally absorbing termination material**.
They are now fixed at physically sized **28cm, 32cm and 22cm**
across their respective walls, with a single perimeter drawn in
exactly the same viewport as the FDTD grid.
They function like door/window *acoustic exits*, not simulated space
on the other side of a true hole. This leaves other surfaces reflective
and deliberately breaks some persistent rectangular-room modes, without
increasing the FDTD stencil cost. The material is frequency-independent,
not a measured physical wall. This only changes the ASMR experiment,
not the arcade's original acoustics.

### Head geometries and eardrums

**The default listener is YOUR `listener_v2.svg`** — both mirrored halves
and the large pinna openings, rather than the old simple round head.
Its original **185.348 × 136.859 mm** viewBox maps to exactly
**0.185348 × 0.136859 m** in the FDTD field, centred at (1.000m,1.185m).
The two authored half-paths are stitched only across their central
straight seam. Their contours are static sound-reflecting SOLID cells.

The **default V2** probes have been moved **outward into the real
open pinna recesses**, at **(0.938m, 1.188m)** and **(1.062m, 1.188m)**, **12.4cm apart**. The prior ±4 cm probes were literally
inside the V2 SVG solid; the program cut artificial ear passages to
reach them. V2 no longer cuts those passages: the probes are in
SVG-authored air, and the GPU sampler requires virtually its entire
bilinear stencil to remain in air. This is an **outer-recess microphone
experiment**, not the same as an 8 cm eardrum pair.

In the **newly redrawn central-cavity**
`listener_with_ear_channel.svg` mode, the microphones are now only
**4.2cm apart** (±21mm from listener center), now **3mm above**
the head's horizontal centerline (raised 10mm after feedback).
Their physical positions are approximately
**(0.979m, 1.188m)** and **(1.021m, 1.188m)**. This new placement
supersedes the earlier 8cm experimental ear-channel separation; it
is a sound-position probe configuration, not a universal anatomical
measurement. At too
coarse a resolution, the program refuses a microphone position that
cannot sample air reliably **or whose sampled air is isolated from the
outside room**. User-edited SVG necks can seal when rasterized at
5mm or another dx; press Q for a finer grid or inspect the SVG path.
We do not delete material to fake a connected ear channel. **H**
advances deterministically V2 → CHANNEL → SIMPLE: no more silent
substitution with the simple SVG. If CHANNEL fails at the selected
grid size, the program tests the gentle square-room grid and chooses
it if it connects. Otherwise the head remains unchanged and a clear
error is printed. The terminal's **[SVG LOAD]** line prints the
exact SVG path and dimensions, and **[H] ACTIVE** prints which model
actually feeds CUDA.

Press **H** in live mode to cycle the alternative geometries:
`listener_v2.svg` (default), `listener_with_ear_channel.svg`, and
`listener.svg`. The previous simple listener uses the experimental
8 cm + 0.8×2 cm ear-canal allowance, giving **9.6 cm outer-recess mic
spacing**. The inner-cavity channel mode now uses the 4.2cm target.

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

When **V** is enabled, the listener mask is the **actual pixelated
acoustic material**, now positioned in the very same aspect-correct
viewport as all microphone markers and the smooth SVG. Press **V**
again to see the authored high-resolution SVG. CPU tests independently
verify the SVG-to-material occupancy at physical metre coordinates,
including the actual GPU image-lookup Y inversion at each zoom.
The authored source is loaded from the current checkout's
`asmr_shapes/` first; a same-named older file under HOME can no
longer unexpectedly replace the current drawing. The [SVG LOAD]
messages print the path actually opened.

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
| Grid cell | **5.0 mm** | **2.0 mm** |
| Acoustic grid X × Y | **400 × 711** | **1000 × 1778** |
| Wave steps/s (CFL) | ~97k | ~243k |
| Ten-cells/wavelength reference | ~6.86 kHz | ~17.15 kHz |
| Source low-pass | **6.0 kHz** | **16 kHz** |
| Sound capture | two FDTD microphones | same two microphones |

Press **Q** to cycle 5 / 4.5 / 3.75 / 3 / 2.5 / 6 / 8 / 10mm live
resolutions (unsupported coarse-grid mic placements are skipped).
The default aims to free GPU time rather than simulate a physically
smaller room. The previous 3.75mm / 8.5kHz preview is still available
as a more demanding option. The offline grid remains independent of live resolution and now
uses **2mm / 16kHz** to preserve the **old offline cell budget**, too.

Every five seconds the terminal prints `[METRES]` with the actual
listener, mouse target, smoothed acoustic head, mouth, left/right
source-to-microphone distances and predicted flight times at **343m/s**.
At 1× the grid's physical width/height are 2.0m × 3.5556m;
gentle zoom and the final square use the actual cropped dimensions
printed by [METRES]. **A true 2D
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
| H | V2 → CHANNEL → SIMPLE, never silently substitute an invalid SVG |
| Q | Cycle real-time acoustic grid resolution |
| Z | Gentle FDTD zoom: **1× / 1.08× / 1.16× / 1.30× / square / back to 1×** |
| F5 / 5 | Render and then **loop exactly 5 s** |
| F6 / 1 | Render and then **loop exactly 10 s** |
| F7 / 2 | Render and then **loop exactly 20 s** |
| P | Restart the most recently completed replay loop |
| **Space** | Stop playback and return to live movement/recording |
| D | Toggle in-solver FDTD test chirp even without MP3 |
| T | Toggle live FDTD audio preview |
| L | Reload the input selected by `--mp3` (or default `~/test.mp3`) |
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
changing the 2mm FDTD simulation, 48kHz stereo buffer, or MP3 input.
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
- Outer walls now reflect strongly (R=0.82); three visible teal
  boundary slots are modeled as absorbing impedances (R=0.035), not
  actual wave propagation outside a geometrically cut aperture.
  This is an approximate, frequency-independent boundary condition.
- The GPU load must be benchmarked in actual runtime, not inferred
  from its board's wattage. The 2mm 9:16 offline grid preserves
  the old square grid's ~1.78 million cells rather than growing to
  ~3.16 million cells. Precomputation may still take much longer
  than playback, but the finished audio plays at normal speed.
- Use low headphone gain during initial tests. Numerical code
  compilation and offline CPU geometry checks cannot verify physical
  loudness or your subjective ASMR response.

The ASMR hypothesis paper motivates exploring proximity and sound
movement but **does not establish** that any particular motion will
induce ASMR:

https://doi.org/10.3389/fnbeh.2025.1688172
