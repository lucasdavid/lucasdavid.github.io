# lucasdavid.github.io

Code for personal web page and blog.

## Development

Plugins used:
```rb
group :jekyll_plugins do
   gem 'jekyll-sitemap'
   gem 'jekyll-paginate'
   gem 'jekyll-feed'
   gem 'jekyll-github-metadata'
   gem 'jekyll-compress-images'
   gem 'jekyll-archives'
   gem 'jekyll-toc'
   gem 'jekyll-scholar'
end
```

Building with docker:
```
docker build -t ldavid/lucasdavid.github.io .
docker run -v $(pwd):/site -p 4000 \
       ldavid/lucasdavid.github.io \
       bundle exec jekyll serve \
       -H 0.0.0.0 -P 4000 \
       --drafts \
       --force_polling
```

Deployment works through GitHub actions,
and builds in the gh-pages branch.
