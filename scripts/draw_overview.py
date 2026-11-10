# scripts/draw_overview.py
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
SAMPLES = OUT / "samples"
FRAME_DIR = Path("Dataset/Resized")
START = 1
N_FRAMES = 4


def win_path(p: Path) -> str:
    return str(p)


def resolve_frame(index: int) -> Path:
    for name in (
        f"frame_{index:05d}.jpg",
        f"frame_{index:05d}.jpeg",
        f"frame_{index:05d}.png",
        f"frame_{index}.jpg",
    ):
        p = FRAME_DIR / name
        if p.exists():
            return p
    raise SystemExit(f"Missing frame {index} in {FRAME_DIR}")


def save_frame(src: Path, dst: Path, size=(380, 240)):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    target = size[0] / size[1]
    if w / h > target:
        nw = int(h * target)
        left = (w - nw) // 2
        im = im.crop((left, 0, left + nw, h))
    else:
        nh = int(w / target)
        top = (h - nh) // 2
        im = im.crop((0, top, w, top + nh))
    im.resize(size, Image.Resampling.LANCZOS).save(dst, "JPEG", quality=90)


def augmentations(im: Image.Image):
    flip = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    arr = np.asarray(im).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    noisy = np.clip(arr + rng.normal(0, np.sqrt(0.01), arr.shape), 0, 1)
    noise = Image.fromarray((noisy * 255).astype(np.uint8), "RGB")

    w, h = im.size
    cw, ch = int(w * 0.75), int(h * 0.75)
    left, top = int(w * 0.18), int(h * 0.08)
    crop = im.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)
    avg = im.filter(ImageFilter.BoxBlur(radius=max(1.8, w / 260)))
    gauss = im.filter(ImageFilter.GaussianBlur(radius=max(1.5, w / 280)))

    masked = im.copy()
    md = ImageDraw.Draw(masked)
    mh = max(8, int(round(h * 32 / 224)))
    mw = max(8, int(round(w * 32 / 224)))
    for fx, fy in ((0.12, 0.15), (0.42, 0.38), (0.66, 0.12), (0.28, 0.60)):
        x, y = int(fx * (w - mw)), int(fy * (h - mh))
        md.rectangle([x, y, x + mw, y + mh], fill=(0, 0, 0))

    return {"Flip": flip, "Noise": noise, "Crop": crop, "Avg": avg, "Gauss": gauss, "Mask": masked}


def tikz_body(paths, augs) -> str:
    p0, p1, p2, p3 = paths
    flip, noise, crop, avg, gauss, mask = augs
    return rf"""
\begin{{tikzpicture}}[font=\small, >=stealth, line join=round, line cap=round]
  \definecolor{{cfront}}{{rgb}}{{0.608,0.349,0.714}}
  \definecolor{{cside}}{{rgb}}{{0.424,0.204,0.514}}
  \definecolor{{ctop}}{{rgb}}{{0.961,0.933,0.973}}
  \definecolor{{cgrid}}{{rgb}}{{0.290,0.137,0.353}}

  \node[inner sep=0pt, outer sep=0pt, draw=black!65, line width=0.45pt] at (1.79,0.45) {{\includegraphics[width=1.90cm,height=1.20cm]{{{p0}}}}};
  \node[inner sep=0pt, outer sep=0pt, draw=black!65, line width=0.45pt] at (1.51,0.23) {{\includegraphics[width=1.90cm,height=1.20cm]{{{p1}}}}};
  \node[inner sep=0pt, outer sep=0pt, draw=black!65, line width=0.45pt] at (1.23,0.01) {{\includegraphics[width=1.90cm,height=1.20cm]{{{p2}}}}};
  \node[inner sep=0pt, outer sep=0pt, draw=black!65, line width=0.45pt] at (0.95,-0.21) {{\includegraphics[width=1.90cm,height=1.20cm]{{{p3}}}}};
  \node[font=\footnotesize, align=center, anchor=north] at (1.37,-1.00) {{Input sequence\\$(T \times 3 \times 224 \times 224)$}};

  \draw[rounded corners=2.5pt, fill=yellow!12, draw=black!55, line width=0.6pt]
    (3.40,-1.68) rectangle (8.65,1.82);
  \node[font=\footnotesize, anchor=south, inner sep=2pt] at (6.025,1.82) {{Training augmentation}};
  \coordinate (augwest) at (3.40,0.45);
  \coordinate (augeast) at (8.65,0.45);

  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, inner sep=1pt]below:Flip}}] at (4.55,0.95) {{\includegraphics[width=1.16cm,height=0.74cm]{{{flip}}}}};
  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, inner sep=1pt]below:Noise}}] at (6.025,0.95) {{\includegraphics[width=1.16cm,height=0.74cm]{{{noise}}}}};
  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, inner sep=1pt]below:Crop}}] at (7.50,0.95) {{\includegraphics[width=1.16cm,height=0.74cm]{{{crop}}}}};
  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, align=center, inner sep=0.4pt]below:{{Avg.\\blur}}}}] at (4.55,-0.48) {{\includegraphics[width=1.16cm,height=0.74cm]{{{avg}}}}};
  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, align=center, inner sep=0.4pt]below:{{Gauss.\\blur}}}}] at (6.025,-0.48) {{\includegraphics[width=1.16cm,height=0.74cm]{{{gauss}}}}};
  \node[inner sep=0pt, draw=black!55, line width=0.4pt, label={{[font=\scriptsize, inner sep=1pt]below:Mask}}] at (7.50,-0.48) {{\includegraphics[width=1.16cm,height=0.74cm]{{{mask}}}}};

  \draw[->, thick] (2.82,0.45) -- (augwest);

  \node[draw, rounded corners=2pt, fill=orange!22, align=center, minimum width=3.05cm, minimum height=1.50cm] (vit) at (10.575,0.45) {{MobileViT-Small}};
  \node[font=\footnotesize, anchor=north] at (10.575,-0.42) {{$(T \times 640)$}};
  \draw[->, thick] (augeast) -- (vit.west);

  \begin{{scope}}[shift={{(12.50,0.14)}}]
    \draw[fill=cside, draw=cgrid, line width=0.55pt]
      (3.84,0) -- (4.76,0.18) -- (4.76,0.80) -- (3.84,0.62) -- cycle;
    \draw[fill=ctop, draw=cgrid, line width=0.55pt]
      (0,0.62) -- (3.84,0.62) -- (4.76,0.80) -- (0.92,0.80) -- cycle;
    \draw[fill=cfront, draw=cgrid, line width=0.55pt]
      (0,0) -- (3.84,0) -- (3.84,0.62) -- (0,0.62) -- cycle;
    \foreach \i in {{1,...,7}}{{
      \pgfmathsetmacro{{\x}}{{\i*0.48}}
      \draw[cgrid, line width=0.4pt] (\x,0) -- (\x,0.62);
      \draw[cgrid, line width=0.4pt] (\x,0.62) -- ({{\x+0.92}},0.80);
    }}
    \draw[<->, thin] (0,-0.22) -- (3.84,-0.22);
    \node[font=\footnotesize, below] at (1.92,-0.24) {{$T$}};
    \draw[<->, thin] (-0.03,0.78) -- (0.89,0.96);
    \node[font=\footnotesize, anchor=south] at (0.43,0.98) {{$640$}};
  \end{{scope}}
  \draw[->, thick] (vit.east) -- (12.50,0.45);

  \node[draw, rounded corners=2pt, fill=teal!16, align=center, minimum width=3.05cm, minimum height=1.50cm] (l1) at (14.875,-3.55) {{BiLSTM 1\\$640 \to 512$}};
  \node[font=\footnotesize, anchor=north] at (14.875,-4.48) {{$(T \times 512)$}};
  \draw[->, thick] (17.26,0.45) -- (17.55,0.45) -- (17.55,-3.55) -- (l1.east);

  \node[draw, rounded corners=2pt, fill=teal!28, align=center, minimum width=3.05cm, minimum height=1.50cm] (l2) at (10.855,-3.55) {{BiLSTM 2\\$512 \to 512$}};
  \node[font=\footnotesize, anchor=north] at (10.855,-4.48) {{$(T \times 512)$}};
  \draw[->, thick] (l1.west) -- (l2.east);

  \draw[fill=cfront, draw=cgrid, line width=0.7pt] (7.74,-4.50) rectangle (8.36,-2.60);
  \draw[cside, line width=0.45pt] (7.74,-4.025) -- (8.36,-4.025);
  \draw[cside, line width=0.45pt] (7.74,-3.550) -- (8.36,-3.550);
  \draw[cside, line width=0.45pt] (7.74,-3.075) -- (8.36,-3.075);
  \node[font=\footnotesize, anchor=north] at (8.05,-4.62) {{$(512)$}};
  \draw[->, thick] (l2.west) -- (8.36,-3.55);

  \node[draw, rounded corners=2pt, fill=red!12, align=center, minimum width=3.10cm, minimum height=1.50cm] (fc) at (5.22,-3.55) {{FC + Dropout\\$512 \to 128 \to 4$}};
  \node[font=\footnotesize, anchor=north] at (5.22,-4.48) {{$(4)$}};
  \draw[->, thick] (7.74,-3.55) -- (fc.east);

  \node[draw, rounded corners=1pt, fill=blue!8, minimum width=2.30cm, minimum height=0.42cm] at (1.55,-2.80) {{Strong}};
  \node[draw, rounded corners=1pt, fill=blue!8, minimum width=2.30cm, minimum height=0.42cm] at (1.55,-3.30) {{Medium}};
  \node[draw, rounded corners=1pt, fill=blue!8, minimum width=2.30cm, minimum height=0.42cm] at (1.55,-3.80) {{Weak}};
  \node[draw, rounded corners=1pt, fill=blue!8, minimum width=2.30cm, minimum height=0.42cm] at (1.55,-4.30) {{None}};
  \node[font=\footnotesize, anchor=north] at (1.55,-4.64) {{Hunger level}};
  \draw[->, thick] (fc.west) -- (2.70,-3.55);
\end{{tikzpicture}}
"""


def write_and_compile(body: str):
    (OUT / "overview.tex").write_text(body, encoding="utf-8")
    standalone = OUT / "overview_standalone.tex"
    standalone.write_text(
        "\\documentclass[tikz,border=10pt]{standalone}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(OUT), standalone.name],
        cwd=OUT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log = OUT / "overview_standalone.log"
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            shown = [ln for ln in lines if ln.startswith("!") or "not found" in ln.lower()]
            print("\n".join(shown[:30] or lines[-25:]))
        else:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
        raise SystemExit(1)
    pdf_src = OUT / "overview_standalone.pdf"
    pdf_dst = OUT / "overview.pdf"
    if pdf_src.exists():
        pdf_src.replace(pdf_dst)
    print("PDF:", win_path(pdf_dst))


def main():
    global FRAME_DIR, START, N_FRAMES
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True, help="Folder of frame_XXXXXX.jpg")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--n-frames", type=int, default=4)
    args = parser.parse_args()
    FRAME_DIR = args.frames
    START = args.start
    N_FRAMES = args.n_frames

    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    chosen = []
    index = START
    while len(chosen) < N_FRAMES:
        if index > START + 400:
            raise SystemExit(f"Could not find {N_FRAMES} frames from {START}")
        chosen.append(resolve_frame(index))
        index += 1

    rel_paths = []
    for i, src in enumerate(chosen, start=1):
        dst = SAMPLES / f"frame_{i}.jpg"
        save_frame(src, dst)
        rel_paths.append(f"samples/frame_{i}.jpg")
        print(f"Sample {i}: {src}")

    base = Image.open(chosen[0]).convert("RGB")
    aug_names = []
    for name, im in augmentations(base).items():
        dst = SAMPLES / f"aug_{name.lower()}.jpg"
        im.resize((280, 176), Image.Resampling.LANCZOS).save(dst, "JPEG", quality=92)
        aug_names.append(f"samples/aug_{name.lower()}.jpg")

    write_and_compile(tikz_body(rel_paths, aug_names))


if __name__ == "__main__":
    main()
