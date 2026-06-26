#!/usr/bin/env python3

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


PAGE_URL = (
    "https://deepblue.lib.umich.edu/data/concern/data_sets/bn999738r"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "alice_eeg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANTED = re.compile(
    r"^(?:"
    r"S\d{2}\.(?:eeg|vhdr|vmrk)"
    r"|README\.txt"
    r"|LICENSE\.txt"
    r"|audio\.zip"
    r"|proc\.zip"
    r"|datasets\.mat"
    r"|comprehension-questions\.doc"
    r"|comprehension-scores\.txt"
    r"|easycapM10-acti61_elec\.sfp"
    r"|AliceChapterOne-EEG\.csv"
    r")$"
)

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/145 Safari/537.36"
        ),
        # Facilita la reanudación mediante Range.
        "Accept-Encoding": "identity",
    }
)

print(f"Consultando {PAGE_URL}", flush=True)
page = session.get(PAGE_URL, timeout=60)
page.raise_for_status()

soup = BeautifulSoup(page.text, "html.parser")

links = {}
for anchor in soup.select("a[href]"):
    filename = anchor.get_text(" ", strip=True)
    if WANTED.fullmatch(filename):
        links[filename] = urljoin(PAGE_URL, anchor["href"])

print(f"Archivos encontrados: {len(links)}", flush=True)

if len(links) < 150:
    raise RuntimeError(
        "No se ha encontrado la lista completa de archivos. "
        "Deep Blue puede haber cambiado el formato de la página."
    )


def download_file(filename: str, url: str) -> None:
    destination = OUT_DIR / filename
    downloaded = destination.stat().st_size if destination.exists() else 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    print(
        f"\n{filename}: comenzando desde {downloaded / 1024**2:.1f} MiB",
        flush=True,
    )

    with session.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 600),
    ) as response:
        if response.status_code == 416:
            print(f"{filename}: ya está completo", flush=True)
            return

        if downloaded > 0 and response.status_code == 206:
            mode = "ab"
            completed = downloaded
        else:
            # El servidor ignoró Range: reiniciar correctamente el archivo.
            mode = "wb"
            completed = 0

        response.raise_for_status()

        remaining = int(response.headers.get("Content-Length", 0))
        total = completed + remaining
        last_report = time.monotonic()

        with destination.open(mode) as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue

                output.write(chunk)
                completed += len(chunk)

                now = time.monotonic()
                if now - last_report >= 5:
                    if total > 0:
                        percentage = 100.0 * completed / total
                        print(
                            f"{filename}: "
                            f"{completed / 1024**2:.1f}/"
                            f"{total / 1024**2:.1f} MiB "
                            f"({percentage:.1f} %)",
                            flush=True,
                        )
                    else:
                        print(
                            f"{filename}: {completed / 1024**2:.1f} MiB",
                            flush=True,
                        )
                    last_report = now

    print(f"{filename}: terminado", flush=True)


def sort_key(item):
    filename = item[0]
    match = re.match(r"S(\d{2})\.(eeg|vhdr|vmrk)$", filename)
    if match:
        extension_order = {"vhdr": 0, "vmrk": 1, "eeg": 2}
        return 0, int(match.group(1)), extension_order[match.group(2)]
    return 1, filename


for filename, url in sorted(links.items(), key=sort_key):
    download_file(filename, url)

counts = {
    extension: len(list(OUT_DIR.glob(f"S??.{extension}")))
    for extension in ("eeg", "vhdr", "vmrk")
}

print(f"\nRecuento final: {counts}", flush=True)

if counts != {"eeg": 49, "vhdr": 49, "vmrk": 49}:
    raise RuntimeError(f"Descarga incompleta: {counts}")

print(f"Dataset descargado correctamente en {OUT_DIR}", flush=True)
