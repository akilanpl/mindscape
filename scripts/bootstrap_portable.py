"""Create a new isolated CPU development environment; never replace an existing one."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=Path('.venv-portable'))
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or newer is required')
    root = Path(__file__).resolve().parents[1]
    target = args.env.resolve()
    if target.exists():
        parser.error('Environment already exists; select a new path')
    venv.EnvBuilder(with_pip=True).create(target)
    python = target / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    subprocess.run([str(python), '-m', 'pip', '--no-cache-dir', 'install', '-r',
                    str(root / 'requirements-cpu-dev.lock.txt')], check=True)
    subprocess.run([str(python), '-m', 'pip', '--no-cache-dir', 'install',
                    '--no-build-isolation', '--no-deps', '-e', str(root)], check=True)
    print('CPU development environment:', target)

if __name__ == '__main__':
    main()
