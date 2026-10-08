import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mindscape.core.schema import Observation
from mindscape.models.learned import LearnedModel, MindscapeModel


def main():
    parser = argparse.ArgumentParser(description='Execute a trained policy and record actual environment feedback')
    parser.add_argument('left', type=int)
    parser.add_argument('right', type=int)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--unmasked', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = LearnedModel.load(args.checkpoint)
    view = {'problem': f'{args.left} × {args.right}', 'environment': 'integer_multiplication',
            'observation': asdict(Observation((args.left, args.right)))}
    if isinstance(model, MindscapeModel):
        model.mask = not args.unmasked
    prediction = model.predict(view)
    payload = {'prediction': asdict(prediction), 'episode': getattr(model, 'last_episode', None)}
    text = json.dumps(payload, indent=2)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
