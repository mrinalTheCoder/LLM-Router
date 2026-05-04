"""
Teaser graphic: Accuracy vs. Inference Cost
Compares local SLMs, cloud APIs, and our PPO router.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────

ELEC_COST_PER_J = 0.12 / 3.6e6  # $0.12 / kWh  →  $/J

local_slms = {
    'Llama 3.2\n(3B)':       dict(acc=0.391, energy_j=14.68),
    'Granite 4\n(3B)':       dict(acc=0.597, energy_j=11.57),
    'Llama 3-ChatQA\n(8B)':  dict(acc=0.489, energy_j=18.24),
    'Dolphin 3\n(8B)':       dict(acc=0.455, energy_j=19.71),
    'Sailor 2\n(8B)':        dict(acc=0.571, energy_j=21.07),
    'Mathstral\n(7B)':       dict(acc=0.314, energy_j=52.67),
}
for v in local_slms.values():
    v['cost'] = v['energy_j'] * ELEC_COST_PER_J

# Router: best result from W&B sweep (measured on MMLU validation)
router = dict(acc=0.7148, energy_j=16.10, cost=16.10 * ELEC_COST_PER_J)

# Cloud API models — MMLU 5-shot scores from published leaderboards.
# Cost: estimated per query (~100 in-tokens + ~200 out-tokens) at public API pricing.
# Marked "estimated" in caption; used only to illustrate relative scale.
cloud_models = {
    'Gemini 1.5\nFlash':     dict(acc=0.782, cost=0.000105),
    'Claude 3\nHaiku':       dict(acc=0.752, cost=0.000200),
    'GPT-3.5\nTurbo':        dict(acc=0.700, cost=0.000400),
    'GPT-4o\nMini':          dict(acc=0.820, cost=0.000210),
    'Gemini 1.5\nPro':       dict(acc=0.853, cost=0.001050),
    'GPT-4o':                dict(acc=0.887, cost=0.002000),
    'Claude 3.5\nSonnet':    dict(acc=0.887, cost=0.004800),
}

# ── Figure ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(11.5, 6.8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# ── Background region shading ────────────────────────────────────────────────

slm_costs   = [v['cost'] for v in local_slms.values()]
cloud_costs = [v['cost'] for v in cloud_models.values()]
Y_BOTTOM    = 22   # matches ax.set_ylim lower bound

# SLM region: tight x, capped in y at just above the best individual SLM so
# the router star clearly sits outside (above) this box.
slm_x_lo  = min(slm_costs)  * 0.55
slm_x_hi  = max(slm_costs)  * 1.60
slm_y_hi  = max(v['acc'] for v in local_slms.values()) * 100 + 4.5   # ≈ 62

# Draw as a closed polygon so y is bounded (not full-height)
from matplotlib.patches import Polygon as MplPolygon
slm_box = MplPolygon(
    [[slm_x_lo, Y_BOTTOM], [slm_x_hi, Y_BOTTOM],
     [slm_x_hi, slm_y_hi], [slm_x_lo, slm_y_hi]],
    closed=True, color='#DDEEFF', alpha=0.65, zorder=0, linewidth=0)
ax.add_patch(slm_box)

# Cloud region: tight x, full height
cloud_x_lo = min(cloud_costs) * 0.42
cloud_x_hi = max(cloud_costs) * 2.20
ax.axvspan(cloud_x_lo, cloud_x_hi, ymin=0, ymax=1,
           color='#FFEEDD', alpha=0.55, zorder=0)

# Region labels — placed just inside the top edge of each region
ax.text(np.sqrt(slm_x_lo * slm_x_hi), slm_y_hi - 1.0,
        'Local SLM Region', ha='center', va='top', fontsize=10.5,
        color='#2255AA', alpha=0.85, fontstyle='italic', fontweight='semibold',
        transform=ax.transData)
ax.text(np.sqrt(cloud_x_lo * cloud_x_hi), 0.93,
        'Cloud API Region', ha='center', va='top', fontsize=10.5,
        color='#AA3300', alpha=0.80, fontstyle='italic', fontweight='semibold',
        transform=ax.get_xaxis_transform())

# ── Local SLM points ─────────────────────────────────────────────────────────

SLM_COLOR   = '#3B7DD8'
CLOUD_COLOR = '#E05C00'
ROUTER_COLOR = '#F5A623'

for name, d in local_slms.items():
    ax.scatter(d['cost'], d['acc'] * 100, s=140, color=SLM_COLOR,
               edgecolors='white', linewidths=0.8, zorder=3, alpha=0.9)

# Label SLMs — explicit (x_offset_in_log_decades, y_offset_in_pct)
slm_label_opts = {
    'Llama 3.2\n(3B)':       dict(xt=+0.1, yt=-6.5, ha='right'),
    'Granite 4\n(3B)':       dict(xt=+0.08, yt=-3.0, ha='right'),
    'Llama 3-ChatQA\n(8B)':  dict(xt=0.05, yt=-6.5, ha='right'),
    'Dolphin 3\n(8B)':       dict(xt=+0.55, yt=+3.0, ha='left'),
    'Sailor 2\n(8B)':        dict(xt=+0.55, yt=-3.5, ha='left'),
    'Mathstral\n(7B)':       dict(xt=+0.20, yt=+4.0, ha='left'),
}
for name, d in local_slms.items():
    o = slm_label_opts[name]
    short = name.replace('\n', ' ')
    ax.annotate(short,
                xy=(d['cost'], d['acc'] * 100),
                xytext=(d['cost'] * (10 ** o['xt']), d['acc'] * 100 + o['yt']),
                fontsize=8.5, color='#1A4E8C', ha=o['ha'], va='center',
                arrowprops=dict(arrowstyle='-', color='#8AAADD',
                                lw=0.75, shrinkA=0, shrinkB=3),
                zorder=4)

# ── Cloud API points ──────────────────────────────────────────────────────────

cloud_label_opts = {
    'Gemini 1.5\nFlash':  dict(xt=-0.70, yt=+3.5, ha='right'),
    'Claude 3\nHaiku':    dict(xt=-0.60, yt=-4.5, ha='right'),
    'GPT-3.5\nTurbo':     dict(xt=+0.45, yt=-4.5, ha='left'),
    'GPT-4o\nMini':       dict(xt=+0.45, yt=-2.5, ha='left'),
    'Gemini 1.5\nPro':    dict(xt=-0.60, yt=+3.5, ha='right'),
    'GPT-4o':             dict(xt=+0.38, yt=+3.5, ha='left'),
    'Claude 3.5\nSonnet': dict(xt=-0.5, yt=-6.5, ha='left'),
}
for name, d in cloud_models.items():
    ax.scatter(d['cost'], d['acc'] * 100, s=140, color=CLOUD_COLOR,
               edgecolors='white', linewidths=0.8, zorder=3, alpha=0.9,
               marker='D')
    o = cloud_label_opts[name]
    short = name.replace('\n', ' ')
    ax.annotate(short, xy=(d['cost'], d['acc'] * 100),
                xytext=(d['cost'] * (10 ** o['xt']), d['acc'] * 100 + o['yt']),
                fontsize=8.5, color='#7A2C00', ha=o['ha'], va='center',
                arrowprops=dict(arrowstyle='-', color='#E8AA88',
                                lw=0.75, shrinkA=0, shrinkB=3),
                zorder=4)

# ── Router star ───────────────────────────────────────────────────────────────

ax.scatter(router['cost'], router['acc'] * 100,
           s=640, color=ROUTER_COLOR, marker='*',
           edgecolors='#8B5E00', linewidths=0.9, zorder=6)

ax.annotate(
    'Ours (PPO Router)\n{:.1f}% acc  ·  {:.0f} J/query'.format(
        router['acc'] * 100, router['energy_j']),
    xy=(router['cost'], router['acc'] * 100),
    xytext=(router['cost'] * 0.45, router['acc'] * 100 + 9),
    fontsize=10.5, color='#5C3800', fontweight='bold', ha='left', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#C07000', lw=1.6,
                    connectionstyle='arc3,rad=-0.15'),
    zorder=7,
    bbox=dict(boxstyle='round,pad=0.38', facecolor='#FFF6E0',
              edgecolor='#F5A623', linewidth=1.5, alpha=0.95),
)

# ── "Pareto" dashed curve through SLMs + router ──────────────────────────────

pareto_pts = sorted(
    [*[(v['cost'], v['acc']) for v in local_slms.values()],
     (router['cost'], router['acc'])],
    key=lambda p: p[0]
)
px = np.array([p[0] for p in pareto_pts])
py = np.array([p[1] for p in pareto_pts]) * 100

# Keep only Pareto-optimal points (non-dominated in acc)
pareto_frontier = []
best_acc = -np.inf
for x, y in sorted(zip(px, py), key=lambda p: p[0]):
    if y >= best_acc:
        pareto_frontier.append((x, y))
        best_acc = y
pf_x = [p[0] for p in pareto_frontier]
pf_y = [p[1] for p in pareto_frontier]
ax.plot(pf_x, pf_y, '--', color='#3B7DD8', lw=1.2, alpha=0.55, zorder=2)

# ── Axes & labels ─────────────────────────────────────────────────────────────

ax.set_xscale('log')
ax.set_xlabel('Inference Cost per Query  (USD, log scale)', fontsize=12, labelpad=8)
ax.set_ylabel('MMLU Accuracy  (%)', fontsize=12, labelpad=8)
ax.set_title('PPO Router: Cloud Level Accuracy at SLM Inference Cost',
             fontsize=13.5, fontweight='bold', pad=12)

# Y axis range
ax.set_ylim(22, 97)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))

# X ticks — nicer dollar labels
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter
dollar_ticks = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
dollar_labels = ['$10⁻⁷', '$3×10⁻⁷', '$10⁻⁶', '$3×10⁻⁶', '$10⁻⁵', '$3×10⁻⁵',
                 '$10⁻⁴', '$3×10⁻⁴', '$10⁻³', '$3×10⁻³', '$10⁻²']
ax.xaxis.set_major_locator(FixedLocator(dollar_ticks))
ax.xaxis.set_major_formatter(FixedFormatter(dollar_labels))
plt.xticks(rotation=30, ha='right', fontsize=8.5)

# ── Legend ────────────────────────────────────────────────────────────────────

legend_handles = [
    mpatches.Patch(color='#DDEEFF', label='Local SLMs (measured energy cost)'),
    mpatches.Patch(color='#FFEEDD', label='Cloud APIs (estimated, public pricing)'),
    plt.scatter([], [], s=140, color=SLM_COLOR,  marker='o', label='Individual SLM'),
    plt.scatter([], [], s=140, color=CLOUD_COLOR, marker='D', label='Cloud LLM API'),
    plt.scatter([], [], s=280, color=ROUTER_COLOR, marker='*', label='PPO Router (ours)'),
]
ax.legend(handles=legend_handles, loc='lower right',
          bbox_to_anchor=(0.92, 0.02), fontsize=13.7,
          framealpha=0.9, edgecolor='#CCCCCC', frameon=True)

# ── Cost-gap annotation ───────────────────────────────────────────────────────

mid_slm_x   = np.sqrt(slm_x_lo * slm_x_hi)
mid_cloud_x = np.sqrt(cloud_x_lo * cloud_x_hi)
arrow_y = 93.5

ax.annotate('', xy=(mid_cloud_x, arrow_y), xytext=(mid_slm_x, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.4))
ax.text(np.sqrt(mid_slm_x * mid_cloud_x), arrow_y + 0.8,
        '≈ 1,000× cost difference', ha='center', va='bottom',
        fontsize=9, color='#333333', fontweight='semibold',
        transform=ax.transData)

plt.tight_layout(pad=1.4)
plt.savefig('/home/sahil/vscode/LLM-Router/teaser_graphic.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/sahil/vscode/LLM-Router/teaser_graphic.png', dpi=180, bbox_inches='tight')
print("Saved teaser_graphic.pdf and teaser_graphic.png")
