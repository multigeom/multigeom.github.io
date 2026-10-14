# multigeom.github.io

Landing page of the paper *A multi-scale information geometry reveals the structure of mutual information in neural populations*
and of its Python package [`multigeom`](https://pypi.org/project/multigeom/). Its "Documentation & Demos" tab opens the Sphinx documentation of the package, which lives in the package repository (<https://multigeom.github.io/multigeom/>). Live at <https://multigeom.github.io>.

A static site with no build step: open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000     # then open http://localhost:8000
```

```
index.html                 the page
static/css/landing.css     the page's style (light and dark); bulma.min.css and index.css are no longer used
static/js/landing.js       the interactive parts: eigenvector teaser, one-neuron widget, eigenvector explorer, BibTeX copy
static/data/               data of the interactive parts (made by scripts/)
static/img/explorer/       eigenvector sprites of the explorer (made by scripts/make_eigvec_assets.py)
docs/assets/images/        figures of the paper
docs/assets/videos/        videos, eigenvector_walk and metric_field, rendered by scripts/render_landing_videos.py in multigeom-tutorials
scripts/make_tuning_widget.py   one-neuron widget: J, J_t, G, distances and mutual information (checks I = 1/2 <G>)
scripts/make_eigvec_assets.py   teaser and explorer data from the stage-4 outputs of the paper's pipeline
```

This repository is frozen once the paper is published: change only broken links.
