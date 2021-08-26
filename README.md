# lucasdavid.github.io

Code for personal web page and blog.

## Development

Jekyll plugins used:
```
jekyll-sitemap
jekyll-paginate
jekyll-feed
jekyll-github-metadata
jekyll-compress-images
jekyll-archives
jekyll-toc
jekyll-scholar
```

JS/CSS used:
```
Bootstrap
AnchorJS
Bootstrap Icons
Google Fonts
```

Building with docker:
```
docker build -t ldavid/lucasdavid.github.io .
docker run --rm -p 4000:4000 -p 35729:35729 -v $(pwd):/site ldavid/lucasdavid.github.io
```

Note: changes in gems require docker rebuild.

## Deployment

Deployment works through GitHub actions,
and builds in the gh-pages branch.
The [jekyll.yml](.github/workflows/jekyll.yml) workflow takes care of this.
