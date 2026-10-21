"""
Timeline (ported unchanged from ../lunarlander-neat, plus subset=True for chapters): put narrated scenes together into one video with a voice-over track and WebVTT captions.

A video is a list of scenes. Each scene has one or more *beats*. In the narration script
(a voice-text `--kind voiceover` work dir) every beat is one `## <scene>` or `## <scene>.<n>`
section, so `## controller`, `## controller.2`, `## controller.3` are the three beats of the
`controller` scene. A beat lasts as long as its narration (plus a short lead-in and tail) or its
minimum length, whichever is longer. Scenes see the beat start times and time their animation to
the voice.

Without narration (no voice dir, or not exported yet) every beat gets its default length, which
is good for previews and stills.
"""

import json
import os
import re
import subprocess

import numpy as np

LEAD = 0.35      # seconds of silence before a beat's narration starts
TAIL = 0.55      # seconds after it ends, before the next beat
SR = 24000       # voice-over sample rate


class Beats:
    """Beat start times of one scene, relative to the scene start."""

    def __init__(self, durations):
        self.durations = list(durations)
        self.starts = list(np.concatenate([[0.0], np.cumsum(self.durations)[:-1]]))
        self.end = float(sum(self.durations))

    def at(self, k):
        """Start of beat k (0-based); a beat the script doesn't have never starts."""
        return self.starts[k] if k < len(self.starts) else float('inf')

    def index(self, t):
        return max(0, int(np.searchsorted(self.starts, t, side='right') - 1))

    def local(self, t):
        """(beat index, time since that beat started, that beat's length)."""
        k = self.index(t)
        return k, t - self.starts[k], self.durations[k]


class Scene:
    """render(t, beats) -> PIL.Image. `beats` is the default length of each beat (seconds),
    `min_beats` an optional minimum per beat (defaults to `beats` for animations that need time)."""

    def __init__(self, name, render, beats, min_beats=None):
        self.name = name
        self.render = render
        self.default = list(beats)
        self.min = list(min_beats) if min_beats is not None else [0.0] * len(self.default)


def load_narration(voice_dir):
    """{'scene': [(seconds, mp3 path, caption text), ...]} from an exported voiceover work dir."""
    if not voice_dir:
        return {}
    tpath = os.path.join(voice_dir, 'audio', 'voiceover', 'timings.json')
    if not os.path.exists(tpath):
        print(f'  (no narration at {tpath}: using default beat lengths)')
        return {}
    with open(tpath) as fh:
        timings = json.load(fh)
    out = {}
    for t in timings:
        m = re.fullmatch(r'([\w-]+?)(?:\.(\d+))?', t['title'].strip())
        if not m:
            raise ValueError(f'narration section {t["title"]!r} is not "<scene>" or "<scene>.<n>"')
        with open(os.path.join(voice_dir, f'{t["scene"]}.md'), encoding='utf-8') as fh:
            caption = fh.read().strip()
        out.setdefault(m.group(1), []).append(
            (t['seconds'], os.path.join(voice_dir, 'audio', 'voiceover', t['file']), caption))
    return out


def plan(scenes, narration, subset=False):
    """[(scene, Beats, [(abs start, seconds, mp3, caption) per beat])] with each beat's length set.
    subset=True renders a part of the narrated video (a chapter), so unused narration is fine."""
    # Chapter cards ("card_*") are silent by design.
    missing = [s.name for s in scenes if narration and s.name not in narration and not s.name.startswith("card_")]
    extra = [] if subset else [n for n in narration if n not in {s.name for s in scenes}]
    if missing or extra:
        raise ValueError(f'narration and scenes disagree: missing {missing}, unused {extra}')
    out, t0 = [], 0.0
    for s in scenes:
        voice = narration.get(s.name)
        if voice and len(voice) != len(s.default):
            raise ValueError(f'scene {s.name} has {len(s.default)} beats, narration has {len(voice)}')
        if voice:
            durs = [max(m, LEAD + sec + TAIL) for m, (sec, _, _) in zip(s.min, voice)]
        else:
            durs = list(s.default)
        beats = Beats(durs)
        cues = []
        for k, d in enumerate(durs):
            if voice:
                sec, mp3, cap = voice[k]
                cues.append((t0 + beats.starts[k] + LEAD, sec, mp3, cap))
        out.append((s, beats, cues))
        t0 += beats.end
    return out


def _decode(mp3):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', mp3, '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _vtt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{s:06.3f}'


def sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', text) if s]


def write_vtt(timeline, path):
    """One cue per sentence, timed by its share of the beat's characters."""
    cues_out = []
    for _, _, cues in timeline:
        for start, sec, _, cap in cues:
            sents = sentences(cap)
            weights = np.array([len(s) + 8 for s in sents], dtype=float)
            bounds = start + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * sec
            cues_out += list(zip(bounds[:-1], bounds[1:], sents))
    lines = ['WEBVTT', '']
    for k, (a, b, s) in enumerate(cues_out):
        nxt = cues_out[k + 1][0] if k + 1 < len(cues_out) else b + 0.5
        lines += [f'{_vtt_time(a)} --> {_vtt_time(min(b + 0.3, nxt))}', s, '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def transcript(timeline):
    """Narration text per scene, for the web page."""
    return [{'scene': s.name, 'text': ' '.join(c[3] for c in cues)} for s, _, cues in timeline if cues]


def render(scenes, output, voice_dir=None, crf=28, fps=30, size=(1280, 720), fade_frames=8, subset=False):
    """Render the scenes to `output` (H.264, plus AAC voice-over and a .vtt when narrated)."""
    timeline = plan(scenes, load_narration(voice_dir), subset=subset)
    total = sum(b.end for _, b, _ in timeline)
    narrated = any(c for _, _, c in timeline)
    print(f'Rendering {total:.1f}s → {output}' + (' (narrated)' if narrated else ' (silent)'))
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-']
    wav = None
    if narrated:
        track = np.zeros(int((total + 1) * SR), dtype=np.float32)
        for _, _, cues in timeline:
            for start, _, mp3, _ in cues:
                pcm = _decode(mp3)
                i = int(start * SR)
                track[i:i + len(pcm)] += pcm[:len(track) - i]
        wav = os.path.splitext(output)[0] + '.voice.f32'
        track[:int(total * SR)].tofile(wav)
        cmd += ['-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', wav, '-c:a', 'aac', '-b:a', '80k']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-shortest', output]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for scene, beats, _ in timeline:
        n = int(round(beats.end * fps))
        print(f'  {scene.name:<14} {beats.end:5.1f}s  ({len(beats.durations)} beat(s))')
        for i in range(n):
            arr = np.asarray(scene.render(i / fps, beats), dtype=np.float32)
            edge = min(i, n - 1 - i)
            if edge < fade_frames:
                arr *= (edge + 1) / (fade_frames + 1)
            ff.stdin.write(arr.astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    if wav:
        os.remove(wav)
        write_vtt(timeline, os.path.splitext(output)[0] + '.vtt')
    return timeline


def still(scene, beat, frac=0.6, voice_dir=None):
    """A frame from `scene` at `frac` of beat `beat` (beats sized like in the video)."""
    durs = list(scene.default)
    narration = load_narration(voice_dir).get(scene.name) if voice_dir else None
    if narration and len(narration) == len(durs):
        durs = [max(m, LEAD + sec + TAIL) for m, (sec, _, _) in zip(scene.min, narration)]
    beats = Beats(durs)
    return scene.render(beats.starts[beat] + frac * beats.durations[beat], beats)
