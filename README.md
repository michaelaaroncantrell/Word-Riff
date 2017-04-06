# Word-Riff

This repository contains the presentation, jupyter notebook and flask app for my final project at Metis, Word Riff. I have built a [website](http://michaelaaroncantrell.pythonanywhere.com/) showcasing the three products I built. The first is an interactive music recommender that allows the user to decide how experimental she wants the recommendations to be.

![Album to Album](https://github.com/michaelaaroncantrell/Word-Riff/blob/master/readme-images/album-to-album.png)

The second is a free form text recommender that allows the user to describe what she wants to hear in her own words.
![Text to Album](https://github.com/michaelaaroncantrell/Word-Riff/blob/master/readme-images/text-to-album.png)

The last finds the musical love child of two given albums.
![Mashup](https://github.com/michaelaaroncantrell/Word-Riff/blob/master/readme-images/mashup.png)

These products were built using about 170,000 amazon.com user reviews of about 4,000 CDs & Vinyl from 2004-2014, leveraging both the star rating and more importantly the textual reviews. I obtained the data courtesy of J. McAuley, C. Targett, J. Shi, A. van den Hengel. 

The salient tools I used to make the models were python, natural language processing and topic modeling. I used flask and pythonanywhere to make the website, and hosted the data in a MongoDB database. The work was done on an AWS cloud computer.