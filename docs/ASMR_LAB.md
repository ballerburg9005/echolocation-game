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
color the live timbre. The **offline higher-resolution WAV is fully
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
The window is **540 × 960**. Signed-pressure visual readback runs
at up to **30 fps** with a **360 × 640** preview, while the solver
still advances at the physically correct acoustic time step. The
finished 48kHz stereo WAV is precomputed independently of live
scheduling, with 180 × 320 pixel replay snapshots at 24fps.

The live full-size view now uses **approximately 5.28mm cells and
379×674 acoustic cells**, compared with the old 5.00mm
400×711 preview. That is approximately **15% less cell-work per
simulated second** before additional zooming. The full-size
high-quality offline preset similarly starts around **2.11mm**,
with a roughly **947×1685** grid; when zooming, both the offline
and live physical domains shrink with the view and use their own
appropriate dx. Source input always plays at 48kHz stereo; only
the acoustic solver's spatial/temporal resolution changes.

Press **Q** to select coarser or finer base resolution while retaining
the exact active H and Z. The default live input low-pass is now
approximately **5.7kHz** (below the ten-cells-per-wavelength
reference of ~6.5kHz). The live SDL queue targets **160ms**,
and the number of graph batches used to refill it scales with
the active acoustic timestep. This fixes the earlier case in which
a 144-batch cap could leave the output starved at close zooms even
when the GPU was capable of more work. On H/Q/Z changes the output
is paused during the CUDA reconfiguration, refilled with newly
computed audio, then resumed with a 50ms attack fade. These
changes address a credible source of switching-related crackle,
but they do not guarantee your GPU will meet live deadlines.

The crackle is **not yet proven** to be GPU starvation. Compare
`solver/compute` (solver capacity), `near-empty queues` (starvation),
`safety-clipped samples` (output clipping), `earPeak` and the `D`
calibration chirp, then compare live sound to the finished WAV. If
crackling persists offline at normal headphone level, investigate the
moving emitter, sample reconstruction and geometry rather than just
increasing the live buffer or GPU power.

### True acoustic zoom and reflective room walls

Press **Z** to cycle through **1.00× → 1.25× → 1.55× → 2.00× →
SQUARE → 1.00×**. These are *real acoustic-domain* zoom levels, not
a display-only crop. The listener remains at a fixed physical world
coordinate. The first four have a 9:16 aspect ratio and retain the
listener at the lower third; the final view is a **0.75 × 0.75m
physically square room**, centered on the head, shown in a square
viewport within the portrait window. It leaves enough clearance to
orbit the speaker around the listener by 360 degrees.

**This version reduces, rather than increases, live FDTD work when
zooming.** Unlike the previous equal-cell-count zoom, it reduces
the physical domain AND adjusts dx so the approximate grid
cell-updates per acoustic second are as follows:

| Z stage | Physical room | Live dx (default Q) | Relative cell-work |
|---|---|---|---|
| 1.00× | 2.00 × 3.56 m | ~5.28 mm | **85%** |
| 1.25× | 1.60 × 2.84 m | ~4.62 mm | **81%** |
| 1.55× | 1.29 × 2.29 m | ~4.09 mm | **76%** |
| 2.00× | 1.00 × 1.78 m | ~3.58 mm | **68.4%** |
| Square | 0.75 × 0.75 m | ~3.31 mm | **27.2%** |

All percentages compare against the former full-size 400×711,
5.00mm live solver. The second-last stage reduces its previous
72% work target by **5%** (to 68.4%). The square stage now
reduces its previous 68% work target by **60%** (to 27.2%
of the original solver work). These targets derive from
*area/dx³*, not measured GPU power or elapsed kernel times.
The square view is still a true close-up of a 0.75m room,
but its acoustic dx is 3.31mm rather than 2.44mm and its
grid is approximately 226×226 cells. Consequently some
fine-wave detail is sacrificed to meet the substantially
lower compute target.

**H and Z now use a single, predictable navigation state.**
H always advances exactly one listener model, Z advances exactly
one zoom level, and Q cycles grid quality. Each preserves the
other two dimensions. **Shift+H, Shift+Z and Shift+Q reverse the
respective cycle.** The currently selected model and zoom are
visible in the window title. If any combination is invalid, the
attempt is rejected without mutating the selection, and a
visible four-second title message explains why.

The inner-ear channel retains its real small cavity and narrow
SVG-authored throat. At coarse dx, simple center-point rasterizing
sealed that throat and disabled H/Z modes. We now find an actual
connected pathway through transparent pixels of the authored
0.4mm SVG raster and project it as a **one-cell-wide, subgrid
topology-preserving airway** onto the coarser FDTD mask.
The drawing itself is left byte-for-byte unchanged, and no
new straight tunnel is added. This is an approximation: a very
narrow physical canal becomes at least one full coarse FDTD
cell wide, so its transmission is not quantitatively reliable.
**CPU tests verify that both ear-channel microphones can receive
sound through an open path at all five zoom levels.**

Rebuilding CUDA geometry pauses SDL audio until a newly computed
160ms queue is ready; a 50ms signal fade reduces audible reset
clicks. The CUDA batch limit scales with timestep size so
close-up modes can refill their output queue. Offline WAV replay
uses the matching physical geometry and its own finer mesh.

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
**3.9cm apart** (±19.5mm from center), **2mm above** the
head's horizontal centerline (another 1mm downward).
Their physical positions are approximately
**(0.9805m, 1.1872m)** and **(1.0195m, 1.1872m)**. This new placement
supersedes the earlier 8cm experimental ear-channel separation; it
is a sound-position probe configuration, not a universal anatomical
measurement. At too
coarse a resolution, the program refuses a microphone position that
cannot sample air reliably **or whose sampled air is isolated from the
outside room**. User-edited SVG necks can seal when rasterized at
5mm or another dx; press Q for a finer grid or inspect the SVG path.
The unchanged SVG is the source for both high-resolution artwork
and the FDTD mask. The narrow ear channel uses a topology-preserving
subgrid rasterization only when required at coarse dx, with no
fabricated direct tunnel: every reopened FDTD cell must intersect
air from an actually connected transparent SVG path.
All five default-Q zooms support CHANNEL, including the original
full-size room and the square close-up. H/Z never silently substitute
a different head or zoom, and impossible alternative Q settings
provide an explicit visible error message.
The terminal's **[SVG LOAD]** prints the original SVG path opened,
and **[H]/[Z] ACTIVE** print the actual model and room.

Press **H** in live mode to cycle the alternative geometries:
`listener_v2.svg` (default), `listener_with_ear_channel.svg`, and
`listener.svg`. The previous simple listener uses the experimental
8 cm + 0.8×2 cm ear-canal allowance, giving **9.6 cm outer-recess mic
spacing**. The inner-cavity channel mode now uses the 3.9cm target.

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
acoustic material**, in the same aspect-correct viewport as all
microphone markers. The original smooth SVG is lightly drawn over
it as a *co-registration guide*, not as acoustic material.
Press **B** to toggle that translucent authored shape and spot any
residual mismatch; the white centre crosshair and cyan/magenta
microphone markers remain at exact physical coordinates. Press **V**
to hide pressure and see the full-opacity authored SVG. CPU tests independently
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
| Grid cell | **~5.28 mm** | **~2.11 mm** |
| Acoustic grid X × Y | **~379 × 674** | **~947 × 1685** |
| Wave steps/s (CFL) | ~92k | ~230k |
| Ten-cells/wavelength reference | ~6.5 kHz | ~16.3 kHz |
| Source low-pass | **~5.7 kHz** | **16 kHz nominal** |
| Sound capture | two FDTD microphones | same two microphones |

Press **Q** to cycle 5 / 4.5 / 3.75 / 3 / 2.5 / 6 / 8 / 10mm live
resolutions (unsupported coarse-grid mic placements are skipped).
The default aims to free GPU time rather than simulate a physically
smaller room. The previous 3.75mm / 8.5kHz preview is still available
as a more demanding option. Offline uses the corresponding physically cropped room and a separate
high-quality grid derived from the old 2mm baseline. It no longer
retains the exact old cell count at every zoom; this reduction is
intentional for interactive responsiveness.

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
| B | Toggle semi-transparent authored SVG over FDTD head mask to audit alignment |
| 9 / 0 | Decrease/increase wave exposure, not the physical pressure |
| H / Shift+H | Next / previous listener: V2 → CHANNEL → SIMPLE; retains zoom |
| Q / Shift+Q | Next / previous compatible acoustic grid quality |
| Z / Shift+Z | Next / previous real zoom: **1× / 1.25× / 1.55× / 2× / 0.75m square**; retains H |
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
changing the 48kHz stereo output, MP3 input, or actual FDTD physics.
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
  from its board's wattage. This version deliberately shrinks the
  FDTD grid domain as zoom increases: the estimated
  85%/81%/76%/68.4%/27.2% relative cell-update budgets do not include
  CUDA graph launch overhead, visualization transfer time, or driver
  scheduling. Offline processing can still be slower than real time,
  but the completed audio plays at its requested 48kHz sample rate.
- Use low headphone gain during initial tests. Numerical code
  compilation and offline CPU geometry checks cannot verify physical
  loudness or your subjective ASMR response.

The ASMR hypothesis paper motivates exploring proximity and sound
movement but **does not establish** that any particular motion will
induce ASMR:

https://doi.org/10.3389/fnbeh.2025.1688172
