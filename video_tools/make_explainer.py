"""Render the narrated explainer and its chapters into docs/video/, plus stills.

  python video_tools/make_explainer.py runs/baseline                 # full + chapters
  python video_tools/make_explainer.py runs/baseline --only explainer
  python video_tools/make_explainer.py runs/baseline --stills-only   # quick preview, no video

Narration comes from voice/explainer/ (voice-text pipeline). Without it the
videos render silent with default beat lengths, which is fine for previews.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as T  # noqa: E402
import vcommon as C  # noqa: E402
import videos_def as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--voice", default=os.path.join(C.REPO_ROOT, "voice", "explainer"))
    ap.add_argument("--out", default=os.path.join(C.REPO_ROOT, "docs", "video"))
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--only", nargs="*", help="explainer and/or chapter-N")
    ap.add_argument("--stills-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ctx = C.Context(args.run_dir, os.path.join(C.REPO_ROOT, "video_tools", "cache"))
    voice = args.voice if os.path.isdir(args.voice) else None

    full = V.explainer(ctx)
    if args.stills_only:
        sd = os.path.join(args.out, "stills")
        os.makedirs(sd, exist_ok=True)
        for s in full:
            for k in range(len(s.default)):
                T.still(s, k, 0.85, voice).save(os.path.join(sd, f"{s.name}_{k}.jpg"), quality=85)
        print(f"stills -> {sd}")
        return
    meta = {"videos": []}
    want = lambda n: not args.only or n in args.only
    if want("explainer"):
        tl = T.render(full, os.path.join(args.out, "explainer.mp4"), voice, crf=args.crf)
        T.still(full[0], 0, 0.8, voice).save(os.path.join(args.out, "explainer.jpg"), quality=86)
        chapters_at, t0 = [], 0.0
        for s, b, _ in tl:
            chapters_at.append({"scene": s.name, "start": round(t0, 2)})
            t0 += b.end
        meta["videos"].append({"slug": "explainer", "title": "NEAT learns to walk", "seconds": round(t0, 1),
                               "scenes": chapters_at, "transcript": T.transcript(tl)})
    for n in range(1, len(V.CHAPTERS) + 1):
        slug, title, sub, scenes = V.chapter(ctx, n)
        if not want(slug):
            continue
        tl = T.render(scenes, os.path.join(args.out, f"{slug}.mp4"), voice, crf=args.crf, subset=True)
        T.still(scenes[1], 0, 0.85, voice).save(os.path.join(args.out, f"{slug}.jpg"), quality=86)
        meta["videos"].append({"slug": slug, "title": f"{n}. {title}", "sub": sub,
                               "seconds": round(sum(b.end for _, b, _ in tl), 1)})
    prev = os.path.join(args.out, "videos.json")
    if args.only and os.path.exists(prev):
        old = {v["slug"]: v for v in json.load(open(prev))["videos"]}
        old.update({v["slug"]: v for v in meta["videos"]})
        meta["videos"] = sorted(old.values(), key=lambda v: (v["slug"] != "explainer", v["slug"]))
    with open(prev, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"videos -> {args.out}")


if __name__ == "__main__":
    main()
