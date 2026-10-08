"""Plot observed complete cells; separate hardware/precision strata without filling gaps."""
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','work/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/coding/emergency_analysis_v1')
data=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
conditions=('model_only','structured','mindscape_b','mindscape_c')
for ax,c in zip(axes.flat,conditions,strict=True):
    lines=defaultdict(list)
    for cell in data['learning_curve']['cells']:
        if cell['condition']==c and cell['complete']:
            lines[(cell['split'],cell['seed'],cell['device'],cell['precision'])].append(cell)
    for (split,seed,device,precision),cells in sorted(lines.items()):
        cells.sort(key=lambda r:r['budget'])
        ax.plot([r['budget'] for r in cells],[100*r['accuracy'] for r in cells],
                color='#2068aa' if split=='test' else '#ca5b22',
                marker='o' if device=='cpu' else 's',linestyle={11:'-',23:'--',37:':'}[seed],
                label=f'{split} seed {seed} {device}/{precision}',alpha=.8)
    ax.set_title(c);ax.set_xscale('log');ax.set_xticks([10,25,50,100],['10','25','50','100'])
    ax.set_ylim(0,105);ax.grid(alpha=.2);ax.set_xlabel('Unique training tasks');ax.set_ylabel('Hidden task success (%)')
    if lines:
        ax.legend(fontsize=6,loc='best')
fig.suptitle(f"Observed learning cells: {data['learning_curve']['completed']}/1920 episodes; incomplete cells omitted")
fig.tight_layout();fig.savefig(root/'learning_partial.png',dpi=180);plt.close(fig)
locked=data['lockbox']['summary']
if locked:
    fig,ax=plt.subplots(figsize=(10,5))
    positions=list(range(len(locked)))
    scores=[100*r['accuracy'] for r in locked]
    ax.bar(positions,scores,color=['#2068aa' if r['split']=='test' else '#ca5b22' for r in locked])
    ax.errorbar(positions,scores,yerr=[[100*r['accuracy']-100*r['wilson_ci95'][0] for r in locked],
                 [100*r['wilson_ci95'][1]-100*r['accuracy'] for r in locked]],fmt='none',ecolor='black',capsize=3)
    ax.set_xticks(positions,[f"{r['condition']}\n{r['split']}\nn={r['episodes']}/50" for r in locked],fontsize=7)
    ax.set_ylim(0,105);ax.set_ylabel('Hidden task success (%)');ax.set_title('Locked independent tasks; 95% Wilson intervals')
    fig.tight_layout();fig.savefig(root/'locked_results.png',dpi=180);plt.close(fig)
print('Plotted observed evidence only')
