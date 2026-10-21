"""Scene lists: the full explainer and its chapters (rendered from the same narration)."""
import scenes as S
from vcommon import title_card

CHAPTERS = [
    ("chapter-1", "The task", "what the walker senses, does and is scored on",
     lambda ctx: [S.scene_task(ctx), S.scene_senses(ctx), S.scene_reward(ctx)]),
    ("chapter-2", "The network", "a pilot with at least one hidden layer",
     lambda ctx: [S.scene_network(ctx), S.scene_hidden(ctx)]),
    ("chapter-3", "How NEAT evolves it", "mutation, species, fitness and the champion",
     lambda ctx: [S.scene_neat(ctx), S.scene_fitness(ctx), S.scene_hof(ctx)]),
    ("chapter-4", "The training run", "from standing still to walking",
     lambda ctx: [S.scene_curve(ctx), S.scene_stages(ctx), S.scene_growth(ctx), S.scene_species(ctx)]),
    ("chapter-5", "Results", "100 unseen terrains, and the comparison",
     lambda ctx: [S.scene_results(ctx), S.scene_laptop(ctx)]),
]


def explainer(ctx):
    out = [S.scene_title(ctx)]
    for _, _, _, body in CHAPTERS:
        out += body(ctx)
    return out + [S.scene_outro(ctx)]


def chapter(ctx, n):
    slug, title, sub, body = CHAPTERS[n - 1]
    return slug, title, sub, [title_card(f"card_{n}", f"Chapter {n}", title, sub)] + body(ctx)
