FROM bretfisher/jekyll-serve

WORKDIR /site
ADD . /site/

RUN gem update bundler
RUN bundle install --retry 5 --jobs 20

ENTRYPOINT [ "/usr/bin/env" ]
CMD [ "bundle", "exec", "jekyll", "serve", "--force_polling", "-H", "0.0.0.0", "-P", "4000", "--livereload", "--drafts", "--incremental" ]
