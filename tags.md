---
layout: page
title: Tags
permalink: /tags/
---

<div class="tags-page">
  <h1>Browse by Tag</h1>
  
  <div class="tag-cloud">
    {% assign all_tags = "" | split: "" %}
    {% for post in site.posts %}
      {% for tag in post.tags %}
        {% assign all_tags = all_tags | push: tag %}
      {% endfor %}
    {% endfor %}
    
    {% assign all_tags = all_tags | uniq | sort %}
    {% for tag in all_tags %}
      {% assign tag_posts = site.posts | where_exp: "post", "post.tags contains tag" %}
      {% assign font_size = tag_posts.size | times: 4 | plus: 80 %}
      <a href="{{ site.baseurl }}/tags/{{ tag }}" class="tag-link" style="font-size: {{ font_size }}%">
        {{ tag }} ({{ tag_posts.size }})
      </a>
    {% endfor %}
  </div>
</div>