#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import requests
from bs4 import BeautifulSoup
import os

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

DIR = os.path.dirname(os.path.realpath(__file__))

def download(url):
    print(f'downloading {url}')
    painting_name = url.split('/')[-1]
    out_fname = f'paintings/{painting_name}.jpg'
    if os.path.exists(out_fname):
        return

    r = requests.get(url)
    assert r.ok
    soup = BeautifulSoup(r.text, 'lxml')
    imgs = soup.find('div', {'class': 'wiki-layout-artwork-info'}).find('aside').find_all('img')
    assert len(imgs) == 1 # Parsing issue/updated HTML if there are more
    img_url = imgs[0]['src']

    r = requests.get(img_url)
    assert r.ok
    with open(out_fname, 'wb') as f:
        f.write(r.content)

with open(DIR+'/wikiart-urls.txt', 'r') as f:
    urls = [line.strip() for line in f.readlines()]

for url in urls:
    download(url)
