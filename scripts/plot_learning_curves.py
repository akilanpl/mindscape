"""Standalone preliminary charts using ReportLab's standard line-plot library."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics import renderSVG
from reportlab.lib import colors


def main():
    parser = argparse.ArgumentParser(description='Plot measured preliminary learning curves')
    parser.add_argument('results')
    args = parser.parse_args()
    folder = Path(args.results)
    records = json.loads((folder / 'curves.json').read_text())
    for split in ['test', 'ood_test']:
        grouped = defaultdict(list)
        for r in records:
            if r['split'] == split:
                grouped[r['model'], r['budget']].append(r['accuracy'])
        drawing = Drawing(700, 380)
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 65, 75, 560, 230
        models = ['baseline', 'mindscape', 'mindscape_unmasked']
        chart.data = [[(n, statistics.mean(grouped[m, n])) for n in sorted({r['budget'] for r in records})] for m in models]
        chart.xValueAxis.valueMin = 0
        chart.xValueAxis.valueMax = max(r['budget'] for r in records)
        chart.xValueAxis.valueSteps = sorted({r['budget'] for r in records})
        chart.yValueAxis.valueMin = 0
        chart.yValueAxis.valueMax = 1.05
        chart.yValueAxis.valueSteps = [0, .25, .5, .75, 1]
        palette = [colors.darkblue, colors.darkgreen, colors.darkorange]
        for i, color in enumerate(palette):
            chart.lines[i].strokeColor = color
            chart.lines[i].strokeWidth = 2
            drawing.add(Line(75 + i * 210, 45, 95 + i * 210, 45, strokeColor=color, strokeWidth=2))
            drawing.add(String(100 + i * 210, 41, models[i], fontSize=10))
        drawing.add(chart)
        drawing.add(String(65, 345, f'PRELIMINARY {split}: accuracy vs training examples', fontSize=15))
        drawing.add(String(65, 322, 'Masked policy is forced by singleton actions; curves do not establish superiority.', fontSize=10))
        drawing.add(String(235, 15, '50: mean of 3 seeds; 100/250: one seed. Raw values in curves.json.', fontSize=9))
        renderSVG.drawToFile(drawing, str(folder / f'{split}_accuracy.svg'))
    print(folder)


if __name__ == '__main__':
    main()
