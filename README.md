---
author:
- Miriam Amin
bibliography:
- 'CH\_Vol2\_Humor\_Detection.bib'
date: 'WS 2019/20'
title: |
     Joke detection with neural networks\
    Project Exposé
---

Introduction
============

Humor is a fundamental property of humans. Although scholars are
analyzing and studying humor since the Ancient Times, it is until today
not completely understood. In contrast to other NLP-related problems,
the computational treatment of humor is far behind.

Former research in computational humor was mainly carried out on Humor
Generation and Humor Detection. As I showed in earlier work
[@aminComputationalHumorAutomatic2019], none of the humor generators
presented so far were able to produce human-like humor. From my
investigations I concluded two approaches which seemed promising for the
advancement of joke generators – a generative and a restrictive
approach. A generative approach to humor generation would aim at
exclusively producing humorous output by preselecting suitable topics to
joke about. A restrictive approach on the other hand would consist of
two systems: A system that produces texts with structural features of
jokes and a second humor detection system that works as a filter letting
only the humorous texts pass. One approach for such a filter would be a
neural network for text classification with the target classes `joke`
and `no joke`.

The aim of this project is to assess the feasability of current neural
network architectures for text classification for the application as
such a joke detector. In the following I will briefly present related
work and earlier systems for joke detection. I will proceed by outlining
the intended method and the data set that will be used for training the
neural network.

[@aminComputationalHumorAutomatic2019]

Related Work
------------

Data Set
--------

Take subset of my joke dataset as positive examples

Create jokes with gpt2-simple as negative examples that are similar to
jokes but are not jokes
