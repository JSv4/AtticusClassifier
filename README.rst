This is still a work in progress and is not meant for public use yet.
Leaving the repository public in case anyone stumbles across it and it
saves some time in preparing your own Atticus classifiers. I am planning
to write a blog post / instructions once the performance is slightly better.

I am using LexNLP for its great sentence tokenization functionality. You
could probably get decent performance out of NLTK. Spacy is ok. Currently,
I have been training the classifiers, then using LexNLP to clean and split
sentences / sections / whatever and then running those chunks through Spacy
and checking the category labels.

To get the full (and excellent) Atticus dataset, go here:
https://www.atticusprojectai.org/

Using a BERT-based model, the beta release of the Atticus training set yields
an acceptable (but still not really production-ready) F-score of .735::

    LOSS 	  P  	  R  	  F
    1.093	0.739	0.472	0.576
    1.960	0.763	0.566	0.649
    0.290	0.756	0.661	0.706
    0.985	0.764	0.683	0.721
    1.616	0.770	0.681	0.723
    0.517	0.743	0.673	0.706
    1.044	0.754	0.697	0.724
    0.127	0.762	0.728	0.745
    0.542	0.748	0.722	0.735
    0.946	0.756	0.722	0.739
    0.219	0.751	0.720	0.735
    0.551	0.751	0.720	0.735

Training the BERT-based model takes a lot more computing power, and a CUDA-compatible
graphics card is absolutely recommended. Using a Nvidia 1050 Ti, the above training
took about three hours.

Training older, Word2Vec-based models yields less promising results but is much,
much faster to train. Spacy's en_web_core_lg model yields an f-score of around .6.
Using Word2Vec embeddings trained on legal data (such as Law2Vec) yields a slightly
better f-score of around .635. Neither is as good as the BERT-based approach, however.