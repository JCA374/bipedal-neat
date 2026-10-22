"""Build the explainer narration from the run's final numbers.

  python video_tools/narration.py runs/baseline > video_tools/narration/explainer.md

One `## <scene>[.n]` section per beat; beat counts must match scenes.py. Written for
the ear: no symbols, short sentences, numbers as digits (narration-prep spells them).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as C  # noqa: E402


def r(x, step=1):
    return int(round(x / step) * step)


def build(ctx):
    ev = ctx.eval100
    tp = ctx.topo
    last = ctx.stats[-1]
    hours = last["wall"] / 3600
    hours_s = f"{hours:.0f} hours" if hours >= 1.5 else "about an hour and a half" if hours >= 1.2 else "about an hour"
    ms = {m: meta for m, (g, meta) in ctx.milestones.items()}
    first = ms.get("first_positive", {}).get("generation")
    top = [m for m in ("300", "200", "100") if m in ms]
    top_m = top[0] if top else None
    hl = tp["depth"] - 1
    layers_s = "one hidden layer" if hl == 1 else f"{hl} hidden layers"
    falls = r(ev["fell_frac"] * 100)
    mean = r(ev["mean"])
    gap_ppo = r(C.PRIOR["PPO"] - mean)

    # Mid-run setting changes (train.py overrides.json -> run_config.json history).
    import json as _json
    hist = _json.load(open(os.path.join(ctx.dir, "run_config.json"))).get("history", [])
    change = next((h for h in hist if h.get("k_seeds", 3) != 3 or h.get("pop_size", 200) != 200), None)
    penalty = next((h for h in hist if h.get("fall_penalty", 0) > 0), None)
    n_hof = len(ctx.champ_meta.get("held_out_seeds") or [0] * 10)
    n_hof_s = {10: "ten", 20: "twenty", 30: "thirty"}.get(n_hof, str(n_hof))
    analysis = _json.load(open(os.path.join(C.REPO_ROOT, "results", "fall_analysis", os.path.basename(ctx.dir.rstrip("/")),
                                            "summary.json"))) if os.path.exists(os.path.join(
        C.REPO_ROOT, "results", "fall_analysis", os.path.basename(ctx.dir.rstrip("/")), "summary.json")) else None
    cmp_path = os.path.join(ctx.dir, "eval", "compare_v1.json")
    cmp = _json.load(open(cmp_path)) if os.path.exists(cmp_path) else None

    s = {}
    s["title"] = ("This is a neural network learning to walk. Nobody trained it with gradients. "
                  "It was evolved, weights and shape together, by an algorithm called NEAT.")
    s["task"] = ("This is Bipedal Walker. A box-shaped body sits on two legs, and each leg has a hip and a knee. "
                 "The ground is bumpy, and it is different every time.")
    s["task.2"] = ("At every step, the walker sets four motor torques, one for each joint, "
                   "each somewhere between minus one and plus one.")
    s["task.3"] = ("The goal is to walk to the far end without falling, within one thousand six hundred steps. "
                   "That is about thirty two seconds of simulated time.")
    s["senses"] = ("To decide, the walker senses twenty four numbers. Four describe the body: "
                   "how it is tilted, how fast it spins, and how fast it moves.")
    s["senses.2"] = ("Ten describe the legs: the angle and speed of every joint, "
                     "and whether each foot is touching the ground.")
    s["senses.3"] = ("And ten come from a lidar, a fan of rays that measures how far away the ground is ahead. "
                     "All of them are scaled to a fixed range, so a trained walker is one small file.")
    s["reward"] = ("The score comes from the environment itself. Moving forward earns points, "
                   "about three hundred for walking the whole course.")
    s["reward.2"] = "Every bit of motor effort costs a little, so flailing lowers the score."
    s["reward.3"] = ("And falling is expensive. If the body touches the ground, the episode ends with minus one hundred. "
                     "The task counts as solved at an average of three hundred, over one hundred random terrains.")
    s["network"] = ("The pilot is a small neural network. The twenty four sensor values come in on the left "
                    "and flow through hidden nodes to four outputs on the right.")
    s["network.2"] = ("Here it is while walking. Blue means a positive signal, orange a negative one. "
                      "Thicker lines are stronger connections.")
    s["network.3"] = "The four outputs on the right are the joint torques, sent to the motors at every step."
    s["hidden"] = ("This project had one hard rule: the network must have at least one hidden layer. "
                   "So every network starts as twenty four inputs, eight hidden nodes and four outputs, "
                   "with no shortcut from an input straight to a motor.")
    s["hidden.2"] = ("Evolution could still remove nodes, so the code checks. It counts the hidden nodes on a real "
                     "path from input to output. A network with none gets its fitness cut, and it can never "
                     "be saved as a champion.")
    s["neat"] = ("NEAT starts with a population of two hundred random networks. Each one is described by a genome: "
                 "a list of its nodes and its weighted connections.")
    s["neat.2"] = ("Children differ from their parents by mutation. Most mutations just change a weight. "
                   "Some add a new connection. And some add a whole new node, by splitting a connection in two. "
                   "That is how the networks grow.")
    s["neat.3"] = ("Each generation, the best fifth survive and have children. "
                   "A child is a copy, or a mix of two parents, plus mutations. The two very best pass on unchanged.")
    s["neat.4"] = ("Similar networks are grouped into species, and they mostly compete within their own species. "
                   "That gives a new structure a few generations to prove itself.")
    s["fitness"] = ("How good is a network? Every network walks the same three random terrains, and its fitness is "
                    "the average score. The terrains change every generation, so nobody can memorise them.")
    if change:
        s["fitness"] += f" Later in the run, that went up to {change['k_seeds']} terrains."
    s["fitness.2"] = ("To save time, an episode stops early if the walker makes no real progress for one hundred and "
                      "fifty steps. A network that just stands there does not waste one thousand six hundred steps of compute.")
    s["fitness.3"] = ("But this creates a trap. Standing still scores about minus nine. A brave attempt that falls "
                      "scores minus one hundred. So at first, standing still wins.")
    s["hof"] = ("The best score in each generation is noisy. Three terrains can be lucky, or unlucky.")
    s["hof.2"] = ("So every ten generations, the top five networks walk ten terrains they have never seen, "
                  "with no early stop. The best average becomes the champion. That is the blue line.")
    if n_hof != 10:
        s["hof.2"] += f" Near the end of the run, that test grew to {n_hof_s} terrains."
    if first is not None:
        s["curve"] = (f"Here is the whole run. For the first {r(first, 10)} generations or so, the best networks "
                      "simply stood still. The trap held.")
        s["curve.2"] = (f"Then, around generation {first}, a network found a slow shuffle that moved forward without "
                        "falling. Its score jumped above zero, and the population followed.")
    else:
        s["curve"] = "Here is the whole run. For a long time, the best networks simply stood still. The trap held."
        s["curve.2"] = "Slowly, some networks learned to move forward without falling."
    plateau = ""
    if change:
        plateau = (f" The scores had levelled off, so from generation {change['generation']} each network got "
                   f"{change['k_seeds']} terrains instead of three, and the population grew to {change['pop_size']}.")
    passed = f"The champion passed {top_m} at generation {ms[top_m]['generation']}." if top_m else ""
    if penalty:
        plateau += (f" And from generation {penalty['generation']}, after a look at why it fell, every fall cost an "
                    "extra hundred points during training.")
    final = (f"After {ctx.G} generations, its average on {n_hof_s} unseen terrains is {r(ctx.champ_meta['mean'])}.")
    s["curve.3"] = " ".join(x for x in (passed, plateau.strip(), final) if x)
    s["stages"] = ("Here are the champions side by side, each on a typical terrain: the middle one of its ten test "
                   "terrains. From a random network, to a shuffle, to a real walk.")
    s["stages.2"] = "And this is the final champion."
    s["growth"] = ("While the scores rose, the networks grew. The violet line counts hidden nodes in use, "
                   "and the amber line counts layers from input to output.")
    start_conns = 24 * 8 + 8 * 4
    if tp["hidden_in_use"] >= 8 and tp["enabled_conns"] >= start_conns * 0.8:
        s["growth.2"] = (f"The champion has {tp['hidden_in_use']} hidden nodes in {layers_s}, "
                         f"and {tp['enabled_conns']} connections. It started as eight nodes in one layer. "
                         "Everything else was added by mutation, and kept because it helped.")
    else:
        s["growth.2"] = (f"The champion has {tp['hidden_in_use']} hidden nodes in {layers_s}, and only "
                         f"{tp['enabled_conns']} connections. It started with eight hidden nodes in one layer and "
                         f"{start_conns} connections. Evolution threw most of them away, and stacked what was "
                         "left into a deeper chain.")
    s["species"] = ("This is the number of species. A setting called the compatibility threshold decides how "
                    "different two networks must be to count as separate species, and the code tunes it to aim "
                    "for about ten.")
    s["species.2"] = ("In the first hour, that tuning ran away. neat python never merges species, so the count "
                      "barely fell, and the threshold kept climbing, to thirty eight. It is now kept within limits, "
                      "and the run continued from its checkpoint.")
    import numpy as _np
    q25, q50, q75 = (r(v) for v in _np.percentile(ev["rewards"], [25, 50, 75]))
    s["results"] = (f"The real test is one hundred terrains the champion has never seen. "
                    f"Its average is {mean}. Half of its runs score between {q25} and {q75}.")
    why = ""
    if analysis:
        fwd = r(analysis["fall_forward_share"] * 100)
        why = (f" Almost all falls look the same: {fwd} percent are forward pitch-overs. The walker speeds up, "
               "its body rises, and then it tips onto its nose within about a third of a second.")
    if cmp and penalty:
        better = cmp["diff_ci"][0] > 0
        worse = cmp["diff_ci"][1] < 0
        verdict = ("That made a real difference" if better else "That made things worse" if worse
                   else "On the same one hundred terrains, the difference is within the noise")
        why += (f" Compared with the champion from before the fall penalty, falls went from {cmp['a']['falls']} "
                f"to {cmp['b']['falls']} in one hundred. {verdict}.")
    if falls >= 4:
        s["results.2"] = (f"But it falls in {falls} of the one hundred, and each fall costs a lot." + why +
                          " Falling is still what pulls the average down.")
    else:
        n = {0: "none", 1: "just one", 2: "just two", 3: "three"}[falls]
        s["results.2"] = (f"It falls in {n} of the one hundred." + why + " It is steady, but slow: "
                          "it walks carefully instead of fast, and speed is what earns the last points.")
    if mean >= 300:
        s["results.3"] = ("That beats the solved line of three hundred. For comparison, the gradient based methods on "
                          "this laptop reached about two hundred and seventy with PPO, and just over three hundred with TD3.")
    else:
        s["results.3"] = ("For comparison, gradient based methods on the same laptop reached about two hundred and "
                          f"seventy with PPO, and just over three hundred with TD3. NEAT is {gap_ppo} points behind PPO, "
                          "without computing a single gradient.")
    s["laptop"] = (f"All of this ran on a four core laptop with no graphics card, in {hours_s} of training. "
                   "Every network is evaluated on its own, so evolution spreads easily across the cores.")
    s["outro"] = ("No gradients. No GPU. Just a population of networks, mutating, competing, and growing. "
                  "The code, the clips and the data are on the project page.")
    return s


def to_markdown(s):
    out = ["# NEAT Bipedal Walker explainer", ""]
    for k, v in s.items():
        out += [f"## {k}", "", v, ""]
    return "\n".join(out)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/baseline"
    ctx = C.Context(run, os.path.join(C.REPO_ROOT, "video_tools", "cache"))
    print(to_markdown(build(ctx)))
