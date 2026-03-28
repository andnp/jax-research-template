import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SPRITES_URL = "https://github.com/k4ntz/JAXAtari/releases/download/v0.1/sprites.zip"


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Install JAXAtari sprite assets into the user data directory.")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing sprite files.")
    return parser.parse_args(argv)


def _jaxatari_assets_dir():
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "jaxatari"
    return Path.home() / ".local" / "share" / "jaxatari"


def _install_is_confirmed():
    return os.environ.get("JAXATARI_CONFIRM_OWNERSHIP") == "1"


def _pong_sprite_dir(assets_dir: Path):
    return assets_dir / "sprites" / "pong"


def _download_and_extract(destination: Path):
    with tempfile.TemporaryDirectory(prefix="jaxatari-assets-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "sprites.zip"
        urllib.request.urlretrieve(SPRITES_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_path)
        shutil.copytree(temp_path / "sprites", destination / "sprites", dirs_exist_ok=True)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    if not _install_is_confirmed():
        raise SystemExit("set JAXATARI_CONFIRM_OWNERSHIP=1 to confirm ownership before installing JAXAtari assets")

    assets_dir = _jaxatari_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)

    if _pong_sprite_dir(assets_dir).exists() and not args.force:
        print(f"JAXAtari assets already present at {assets_dir}")
        return 0

    _download_and_extract(assets_dir)

    if not _pong_sprite_dir(assets_dir).exists():
        raise SystemExit(f"expected pong sprite assets in {assets_dir} after installation")

    print(f"Installed JAXAtari assets to {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())