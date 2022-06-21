# lucasdavid.github.io
Code for personal web page and blog.

##### Build with docker
```shell
docker build -t ldavid/lucasdavid.github.io .
docker run --rm -p 4000:4000 -p 35729:35729 -v $(pwd):/site ldavid/lucasdavid.github.io
```
##### Deployment
The [jekyll.yml](.github/workflows/jekyll.yml) workflow takes care of compiling the site,
storing it in the gh-pages branch and deploying it.
