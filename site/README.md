# Latch website

From the release root, preview the static site with:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/site/`. Keep `docs/`, `examples/` and the root README
beside `site/` so their links work. Downloads link directly to the
[v0 GitHub release](https://github.com/DeepakSingh98/latch/releases/tag/v0).
No build step is needed.
