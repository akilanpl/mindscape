"""Readable ablation labels at the lower axis; retains the original figure."""
import argparse,json,math
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib import colors
LABELS={'dream':'Dream','matched_information_control':'Matched policy','no_dream':'No dream','no_environment_interaction':'No real env','no_episodic_memory':'No replay store','no_explicit_state':'No state','no_goal_evaluator':'No goal flag','no_trajectory_supervision_regime_C':'C regime'}
def main():
 p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results);rows=[r for r in json.loads((root/'ablations.json').read_text())['measured'] if r['split']=='test'];out=root/'plots'/'ablation_effects.svg'
 if out.exists():raise ValueError('Refuse overwrite')
 values=[100*r['delta_accuracy'] for r in rows];d=Drawing(900,480);d.add(String(60,445,'IID accuracy change at n50 (percentage points)',fontSize=16));d.add(String(60,422,'Three seeds; controls and regime contrasts have different interpretations.',fontSize=10))
 chart=VerticalBarChart();chart.x=60;chart.y=105;chart.width=800;chart.height=280;chart.data=[values];chart.categoryAxis.categoryNames=[LABELS[r['variant']] for r in rows];chart.categoryAxis.labels.fontSize=9;chart.categoryAxis.labelAxisMode='low';chart.categoryAxis.joinAxisMode='bottom';chart.valueAxis.valueMin=min(0,math.floor(min(values))-1);chart.valueAxis.valueMax=max(1,math.ceil(max(values))+1);chart.bars[0].fillColor=colors.HexColor('#1b9e77');d.add(chart);d.add(String(60,42,'No real env removes grounding by definition; C changes feedback/labels; no goal flag retains external scoring.',fontSize=9));renderSVG.drawToFile(d,str(out));print(out)
if __name__=='__main__':main()
