# multigeom.github.io

Landing page of the paper *A multi-scale information geometry reveals the structure of mutual information in neural populations*
and of its Python package [`multigeom`](https://pypi.org/project/multigeom/). Its "Documentation & Demos" tab opens the Sphinx documentation of the package, which lives in the package repository (<https://multigeom.github.io/multigeom/>). Live at <https://multigeom.github.io>.

A static site with no build step: open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000     # then open http://localhost:8000
```

```
index.html              the page
static/css/             Bulma (vendored, v0.9.4) and the page's own style
docs/assets/images/     figures of the paper
docs/assets/videos/     videos, eigenvector_walk and metric_field, rendered by scripts/render_landing_videos.py in multigeom-tutorials
```

This repository is frozen once the paper is published: change only broken links.
