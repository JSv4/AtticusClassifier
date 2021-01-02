******************************************
Atticus Legal Clause Classifiers for Spacy
******************************************

Introduction
############

The `Atticus Project <https://www.atticusprojectai.org/>`_ was recently announced as an initiative
to, among other things, build a world-class corpus of labelled legal contracts which could be used
to train and/or benchmark text classifiers and question-answering NLP models. Their initial release
contains 200 labelled contracts. I wanted to experiment with the data set and build a working classifier
that I could use on contract data, so I set out to build a simple project to load the dataset, convert it
into a format that Spacy can read, and then train some classifiers to see how the data set performs.
This repository contains the code I used to train classifiers based on 1) Word2Vec embeddings and 2)
a BERT-based transformer model.

Quickstart - Use the Classifier
###############################

If you are in a hurry to test out the classifiers and are not really interested in how they were trained,
you can currently install the classifier directly from a package I'm hosting on my AWS bucket by typing::

    pip install https://jsv4public.s3.amazonaws.com/en_atticus_classifier_bert-0.1.0.tar.gz

This should install Spacy, Spacy-transformers, a BERT model and the classifiers. Once you've installed the
model, you can use it like this::

    import spacy

    nlp = spacy.load('en_atticus_classifier_bert')

    clause = """The Joint Venturers shall maintain adequate books
    and records to be kept of all the Joint Venture activities and affairs
    conducted pursuant to the terms of this Agreement. All direct costs and
    expenses, which shall include any insurance costs in connection with the
    distribution of the Products or operations of the Joint Venture, or if the
    business of the Joint Venture requires additional office facilities than
    those now presently maintained by each Joint Venturer"""

    cats = nlp(clause).cats
    cats = [label for label in cats if cats[label] > .7] # If you want to filter by similarity scores > .7
    print(cats) # Show the categories


As discussed below, the performance of the model is good enough to be interesting,
but currently not good enough to really be production ready. I *think* this is primarily
due to the dataset being relatively small and many clause categories having fewer than 20
examples. I wanted to release this as-is, however so others could experiment. As the Atticus
Project corpus grows, these classifiers should get better. In my experience 50 - 100 examples
is typically a good target to aim for, so doubling or tripling the Atticus Corpus will
hopefully lead to much, much better performance.

Build a Word2Vec-Based Model
############################

I first experimented with using Spacy's OOTB Word2Vec models. This approach was very
quick to train, but the performance was not very good. The f-score was about .6. I also
tried using a different set of word embeddings released as "Law2Vec", and these improved
performance marginally to an F-Score of ~.64. I've included the code to train these models
in Word2VecModelBuilder.py. You can simply run that python script. The default settings
will load Spacy's en_core_web_lg model and embeddings. You can also load the Law2Vec model
if you download the vector file::

    wget -O ~/Downloads/Law2Vec.200d.txt https://archive.org/download/Law2Vec/Law2Vec.200d.txt

Then you can use Spacy to convert this file into a Spacy-compatible model like so::

    mkdir /models
    python -m spacy init-model en /models/Law2VecModel --vectors-loc ~/Downloads/Law2Vec.200d.txt

Then you can change the model argument (per the example above) to '/models/Law2VecModel'.
You probably want to change the output_dir too. Once you've trained a new model, you can
load the trained model with spacy.load(output_dir).

Train a BERT-based Model
########################

Overview
  The transformer models encode a lot more contextual information about words than Word2Vec models,
  so I wanted to see if I could squeeze more performance out of the dataset using BERT. The good
  news was performance increased substantially using a BERT-based model. This is still probably not good enough for use in production, but it's good
  enough to yield some interesting insights, particularly if you set your similarity threshold very
  high.

Training Results
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

Step 1 - Sign Up for Atticus Project Data and Download
  I've included the Atticus CSV in the repository for convenience, but you should go to the
  Atticus Project website and signup there. For one, they would like to collect user and
  contact info for people downloading their dataset. For another, you should go there to make
  sure you get the latest version of their dataset.

Step 2 - Install Python Dependencies and SPACY BERT Model
  First, install Python dependencies (I'm using LexNLP to tokenize test data, you do not
  need it to build the model)::

    pip install lexnlp spacy pip install spacy-transformers==0.5.2 pandas

  Then, download the BERT transformer model::

    !python -m spacy download en_trf_bertbaseuncased_lg

Step 3 - Load Atticus Data and Format for Spacy
  The Atticus dataset is a csv, so we can use Pandas to load and manipulate it. Since
  we're training classifiers and not answering questions, we only care about the columns
  containing text for a given classification. The columns with headers marked "...-Answer"
  are meant for question-answering and we don't want to train on this data. We also don't
  really want the filename column or the document title columns, which are the first and
  second columns respectively. The following function will load our Atticus CSV, filter
  out the ...-Answer cols, the filename col and the document title col. Then, it will
  format the data into Spacy's preferred training format and split the training set into
  two pieces - a training set and an evaluation set. The default is to split the total data
  set so 80% is used for training and 20% is used for evaluation.

  **Code**::

        def load_atticus_data(filepath='/tmp/aok_beta/Final Publication/master_clauses.csv'):

            """
            Load data from the atticus csv (omitting the answer cols as we want to train classifiers
            not question answering).

            Data is returned in the Spacy training format:
                TRAIN_DATA = [
                    ("text1", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}})
                ]

            A list of headers is also returned so you can add these labels. FYI, the Filename and Doc name
            columns are dropped as well.

            """

            # Load csv
            atticus_clauses_df = pd.read_csv(filepath)

            # Do a little post-processing
            data_headers = [h for h in list(atticus_clauses_df.columns) if not "Answer" in h]
            data_headers.pop(0)  # Drop filename col (index 0 for col 1)
            data_headers.pop(0)  # Drop doc name (orig col 2 (index 1) but now first col (index 0))

            training_values = {i: 0 for i in data_headers}
            atticus_clauses_data_df = atticus_clauses_df.loc[:, data_headers]

            train_data = []

            # Iterate over csv to build training data dict
            for header in atticus_clauses_data_df.columns:

                for row in atticus_clauses_data_df[[header]].iterrows():

                    value = row[1][header]

                    if not pd.isnull(value):
                        train_data.append((value, {'cats': {**training_values, header: 1}}))

            return train_data, data_headers


        def create_training_set(train_data=[{}], limit=0, split=0.8):
            """Load data from the Atticus dataset, splitting off a held-out set."""
            random.shuffle(train_data)
            train_data = train_data[-limit:]

            texts, labels = zip(*train_data)
            split = int(len(train_data) * split)

            # Return data in format that matches example here:
            # https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
            return (texts[:split], labels[:split]), (texts[split:], labels[split:])


Step 4 - Build the Model
  *WARNING - running the training takes a looong time, even if you have a CUDA-compatible
  graphics card and it's properly configured in your environment*

  You can just run the BertModelBuilder.py with default settings. On my Nvidia 1050 Ti, it took
  about 3 - 4 hours to run the training. Unless you're adding additional data, I'd suggest you
  just use my pre-built models.

Packaging / Serving Model for Use
#################################

You can follow Spacy's excellent instructions `here <https://spacy.io/api/cli#package>`_
to package up the final model into a tar that can be installed with pip like this::

    pip install local_path_to_tar.tar.gz

I've uploaded the package to my public AWS bucket, and you can install directly from there
like so::

    pip install https://jsv4public.s3.amazonaws.com/en_atticus_classifier_bert-0.1.0.tar.gz

Now you can load it just like this::

    nlp = spacy.load('en_atticus_classifier_bert')

I plan to also upload this to PyPi as well so you can just do something like this::

    pip install atticus_classifiers_spacy (DOESN'T WORK YET)

Another option, is you can load the pickled model in the pre-trained folder::

    import pickle
    import spacy

    nlp = pickle.load(open("/path/to/BertClassifier.pickle", "rb"))

    # Then you can use the spacy object just like normal:
    clause = "Test clause"
    cats = nlp(clause).cats
    cats = [label for label in cats if cats[label] > .7] #If you want to look only at labels with similarity scores over .7
    print(cats)
