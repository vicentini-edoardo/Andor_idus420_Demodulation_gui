# Product

## Register

product

## Users

PhD students and researchers (2-5 people) sharing a single SNOM/Andor iDus 420 setup in a condensed-matter or nano-optics lab. They know the instrument well but have varying familiarity with the software. Sessions are focused and task-driven: cool the camera, set ROI, align demodulation, then acquire or scan. Mental load is high from the physics itself; the UI should remove friction, not add it.

## Product Purpose

Desktop control software for the Andor iDus 420 spectrometer in near-field PL demodulation experiments. Users connect to the camera, configure cooling and frame geometry, inspect live spectra, define ROI-based demodulation parameters, run timed acquisitions, and execute 2-D SNOM raster scans — all from one application. Success means an operator moving from startup to a completed, saved scan without confusion or unplanned interruptions.

## Brand Personality

Competent, exact, unfussy. Research workstation: organised and trustworthy, with enough polish to feel purpose-built rather than cobbled together — but never decorative for its own sake.

## Anti-references

- Flashy SaaS dashboards (animated numbers, gradient KPI cards, hero metrics)
- LabVIEW-style dense-grey cluttered panels with no visual hierarchy
- Consumer apps that hide controls behind non-standard gestures
- Any UI where cosmetics compete with instrument data for attention

## Design Principles

1. **Instrument first** — data and plots take visual priority; controls are subordinate.
2. **State is always visible** — connection, temperature, acquisition running/idle, and errors must be unambiguous at a glance, not buried in a status bar.
3. **Workflow has a direction** — the tab order (Camera → Live → Demod → Acquire → Scan) mirrors the real workflow; the UI should reinforce it, not obscure it.
4. **Density is earned** — compact controls are fine when users know what they're doing; spacing should open up where decisions matter (scan grid, demod settings).
5. **Nothing surprising** — standard Qt affordances, consistent vocabulary, no invented patterns that make a user pause.

## Accessibility & Inclusion

WCAG AA minimum. Colorblind-safe plot palettes (avoid red/green-only encoding). Adequate contrast on status indicators — connection and acquisition states must not rely on color alone.
