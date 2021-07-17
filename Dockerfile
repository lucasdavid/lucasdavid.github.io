FROM bretfisher/jekyll-serve

WORKDIR /site
ADD . /site/

# RUN bundle install --retry 5 --jobs 20
CMD bundle exec jekyll serve --force_polling -H 0.0.0.0 -P 4000 --livereload
