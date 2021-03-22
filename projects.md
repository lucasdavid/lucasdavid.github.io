---
layout: default
title: Projects
---

<h1 class="display-4 mb-4">{{ page.title }}</h1>

## GitHub Repositories

<div class="mt-4 mb-4">
  {% assign repos = site.github.public_repositories | where:'fork', false | sort: 'stargazers_count' | reverse %}
  {% for r in repos limit: 12 %}
  <div class="card border-0 mb-1">
    <div class="card-body p-0 pt-2 pb-2">
      <p class="card-title small mb-1">
        <strong>
          <i data-feather="book" width="12" height="12"></i>
          <a href="{{ r.html_url }}" target="_blank">{{r.name}}</a>
        </strong>
      </p>
      <p class="card-text small mb-1">
        {{r.description}}
      </p>
      <p class="text-muted small mb-0" style="font-size: .7em;">
        <i data-feather="star" width="12" style="margin-top:-2px;"></i> {{r.stargazers_count}}
        Last pushed at {{r.pushed_at | date: "%b %e, %Y" }}
      </p>
    </div>
  </div>
  {% endfor %}
</div>

## Organizations

<div class="mt-4 mb-4">
  <div class="card mb-1 border-0">
    <div class="card-body p-0 pt-2 pb-2">
      <p class="card-title small mb-1">
        <strong>
          <i data-feather="book" width="12" height="12"></i>
          <a href="https://github.com/Comp-UFSCar">Comp-UFSCar</a>
        </strong>
        
      </p>
      <p class="card-text small mb-1">
        Contains a bunch of study material which you can use to help you through your undergrad classes.
      </p>
      <p class="text-muted small mb-0" style="font-size: .7em;">
        <a href="https://github.com/Comp-UFSCar">https://github.com/Comp-UFSCar</a>
      </p>
    </div>
  </div>
</div>