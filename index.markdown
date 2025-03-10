---
layout: home
title: Blogs
---
Welcome to my blogs site. Here I will share my learning about various topics.
<br>
Stay tuned!

<h2 class="post-list-heading">Browse by Tag</h2>
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
    <a href="{{ site.baseurl }}/tags/{{ tag }}" class="tag-link">
      {{ tag }} ({{ tag_posts.size }})
    </a>
  {% endfor %}
</div>