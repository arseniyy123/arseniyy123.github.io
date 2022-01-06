---
title: 'Get Me a Movie'
date: 2022-01-06
featured_image: '/images/getmeamovie.gif'
excerpt: "Don't know what to watch today? Get me a movie will help you"
---

*As part of my MSc degree I had a subject where I had the pleasure to team up with really nice folks to build an app with AI/ML service integrated inside, please find below the result*

## Context

Nowadays, we're really busy with our day-to-day life, tasks and projects, so when we have some time to relax some of us decide to watch some movie. 

But given that today we also have a lot of film content one might get stuck and/or just cannot decide what to watch. We know that finding a movie to watch can be time-consuming and boring and tot all movies are available on all platforms. Platforms like Netflix don’t show the rating of a movie so we can't know how likely we would like some movie.

To tackle this problem we have built **Get me a movie** page, an AI-based app that makes finding a movie to watch easier than ever through a great User Experience

![png](/images/getmeamovie.gif)

## Tech Stack

This page is build with the following technologies:

- **Frontend**: React.js, TailwindCSS, MobX and Figma 
- **Backend**: Python, Poetry, FastAPI and Docker
- **Data & ML**: Python & MongoDB 

## Model

In order to suggest some movies we first have to build a recommendation system. There are different ways to create it, we picked the hybrid way of doing it in other words we combine two main factors for recommendations: **Item based collaborative filtering** and **Content based recommenders**. So, similar movies should have similar ratings and also similar movies should have similar descriptions.

<img src="/images/aggregations.png" width="500" height="600" />

If you have any question reach out to us.

Page is available: [here](http://getmeamovie.com/)

Github project: [here](https://github.com/ADS2021UB/GetMeAMovie/tree/development)