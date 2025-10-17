import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

try:
    # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version as dist_version
except Exception:
    # Backport for Python <3.8
    from importlib_metadata import PackageNotFoundError, version as dist_version  # type: ignore


@dataclass
class Item:
    name: str                # display name
    category: str
    module_names: List[str]  # module import names
    dist_names: List[str]    # PyPI distribution names to try
    version: Optional[str] = None
    installed: bool = False
    source: Optional[str] = None  # "dist" or "module"
    notes: Optional[str] = None


LIBS: List[Item] = [
    # Audio
    Item("librosa", "Audio", ["librosa"], ["librosa"]),
    Item("soundfile", "Audio", ["soundfile"], ["soundfile", "SoundFile"]),
    Item("scipy", "Audio", ["scipy"], ["scipy"]),
    # Image
    Item("opencv-python", "Image", ["cv2"], ["opencv-python", "opencv-contrib-python"]),
    Item("pillow", "Image", ["PIL"], ["Pillow"]),
    Item("scikit-image", "Image", ["skimage"], ["scikit-image"]),
    Item("matplotlib", "Image", ["matplotlib"], ["matplotlib"]),
    # Video
    Item("moviepy", "Video", ["moviepy"], ["moviepy"]),
    # General
    Item("numpy", "General", ["numpy"], ["numpy"]),
    Item("pandas", "General", ["pandas"], ["pandas"]),
    Item("jupyter", "General", [], ["jupyter", "jupyter-core", "notebook", "jupyterlab"]),
]


def get_version_from_dist(names: List[str]) -> Optional[str]:
    for n in names:
        try:
            v = dist_version(n)
            if v:
                return str(v)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def get_version_from_module(names: List[str]) -> Optional[str]:
    import importlib
    for m in names:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, "__version__", None)
            if v:
                return str(v)
            # Pillow keeps version on PIL.__version__
            if m == "PIL":
                v = getattr(mod, "PILLOW_VERSION", None) or getattr(mod, "__version__", None)
                if v:
                    return str(v)
        except Exception:
            continue
    return None


def check_jupyter() -> Optional[str]:
    # Prefer CLI if available
    try:
        out = subprocess.check_output(["jupyter", "--version"], text=True, stderr=subprocess.STDOUT, timeout=10)
        first = out.splitlines()[0].strip()
        return first
    except Exception:
        # Fall back to dist names
        return get_version_from_dist(["jupyter", "jupyter-core", "notebook", "jupyterlab"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check versions of audio, image, video, and general purpose libraries.")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print(f"Python : {platform.python_version()}  ({sys.executable})")
    print(f"OS     : {platform.system()} {platform.release()} on {platform.machine()}\n")

    results: List[Item] = []

    for it in LIBS:
        if it.name == "jupyter":
            v = check_jupyter()
            if v:
                it.installed = True
                it.version = v
                it.source = "cli/dist"
            else:
                it.installed = False
                it.notes = "jupyter CLI or core packages not found"
            results.append(it)
            continue

        v = get_version_from_dist(it.dist_names)
        if v:
            it.installed = True
            it.version = v
            it.source = "dist"
            results.append(it)
            continue

        v = get_version_from_module(it.module_names)
        if v:
            it.installed = True
            it.version = v
            it.source = "module"
        else:
            it.installed = False
            it.notes = "not importable"
        results.append(it)

    if args.json:
        data = [asdict(r) for r in results]
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    # Pretty table
    colw = {"category": 8, "name": 16, "installed": 9, "version": 32, "source": 8, "notes": 26}
    header = f"{'Category':<{colw['category']}} {'Package':<{colw['name']}} {'Installed':<{colw['installed']}} {'Version / Info':<{colw['version']}} {'Source':<{colw['source']}} {'Notes':<{colw['notes']}}"
    print(header)
    print("-" * len(header))
    for r in results:
        ver = r.version or "-"
        src = r.source or "-"
        note = r.notes or "-"
        print(f"{r.category:<{colw['category']}} {r.name:<{colw['name']}} {str(r.installed):<{colw['installed']}} {ver:<{colw['version']}} {src:<{colw['source']}} {note:<{colw['notes']}}")

    # Optional quick smoke tests
    print("\nQuick smoke tests:")
    try:
        import numpy as _np
        _ = (_np.zeros((2, 2)) + 1).sum()
        print("  numpy basic ops: OK")
    except Exception as e:
        print(f"  numpy basic ops: FAIL ({e})")

    try:
        import pandas as _pd
        _ = _pd.DataFrame({"a": [1, 2], "b": [3, 4]}).shape
        print("  pandas DataFrame: OK")
    except Exception as e:
        print(f"  pandas DataFrame: FAIL ({e})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot([0, 1], [0, 1]); plt.close()
        print("  matplotlib plot (Agg): OK")
    except Exception as e:
        print(f"  matplotlib plot: FAIL ({e})")

    try:
        import cv2
        _ = cv2.UMat(1)
        print("  opencv import: OK")
    except Exception as e:
        print(f"  opencv import: FAIL ({e})")

    try:
        import moviepy
        # Try to locate ffmpeg via imageio_ffmpeg if present
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"  moviepy import: OK  ffmpeg: {ff}")
        except Exception:
            print("  moviepy import: OK  ffmpeg not auto-detected")
    except Exception as e:
        print(f"  moviepy import: FAIL ({e})")

    try:
        import librosa, soundfile, scipy  # noqa
        print("  audio stack import: OK")
    except Exception as e:
        print(f"  audio stack import: FAIL ({e})")

    try:
        import skimage, PIL  # noqa
        print("  image stack import: OK")
    except Exception as e:
        print(f"  image stack import: FAIL ({e})")


if __name__ == "__main__":
    main()
