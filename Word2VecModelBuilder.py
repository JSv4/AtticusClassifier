# This relies heavily on the Spacy 2 architecture and pre-trained Word2Vec models. I've tried both the Law2Vec model
# (200-dimension vetors) and Spacy en_core_web_lb (300-dimensions). Law2Vec appears to have a slightly better F-score
# BUT both are pretty bad at the moment. I think this is due to the relatively small number of training examples
# in the initial Atticus dataset. There's only 200 contracts and some clauses in particular have far fewer examples.
# In my experience using commercial systems, 50+ examples is prudent for a Word2Vec approach, so I think doubling or
# tripling the training set would make a huge difference.
#
# I am also going to experiment with using a BERT classifier to try to overcome some of the limitations in the Word2Vec
# Approach. Look at BertClassifier.py

from __future__ import unicode_literals, print_function
import random
import plac
import spacy
from pathlib import Path
from spacy.util import minibatch, compounding
from AtticusUtils.Loaders import load_atticus_data


def create_training_set(train_data=[{}], limit=0, split=0.8):
    # Partition off part of the train data for evaluation
    random.shuffle(train_data)
    train_data = train_data[-limit:]

    texts, labels = zip(*train_data)
    split = int(len(train_data) * split)

    # Return data in format that matches example here:
    # https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
    return (texts[:split], labels[:split]), (texts[split:], labels[split:])

"""
    Note the arguments below. The two that are really most important are the model name and the output_dir. 
    Your trained model will be saved to output_dir and can be loaded from there for use later. The model args
    can be a standard spacy model (default is en_core_web_lg) or you can load a model you've created from custom 
    word embeddings models. I've found the Law2Vec embeddings available here: https://archive.org/details/Law2Vec#reviews
    are pretty nice out of the box and meaningfully outperform the Spacy model. In order to use them, however, first
    you need to convert the txt vector file into a Spacy model like so:
    
     python -m spacy init-model en /output/path/to/NewVectorModel --vectors-loc /path/to/Law2Vec.200d.txt

    Once you've done this, you can pass /output/path/to/NewVectorModel as the model argument below to build classifiers
    on this custom set of word embeddings. My experience so far, however, is the Word2Vec models need more training data 
    than is currently available from the Atticus project's dataset. 

"""
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path),
)
def main(model='en_core_web_lg',
         output_dir='/models/EnCoreLgClassifier',
         n_iter=20, n_texts=2000, init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "ensemble"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # load the Atticus dataset
    print("Loading Atticus Project training data...")
    train_data, data_headers = load_atticus_data()
    (train_texts, train_cats), (dev_texts, dev_cats) = create_training_set(train_data=train_data)
    train_cats = [i['cats'] for i in train_cats]
    dev_cats = [i['cats'] for i in dev_cats]

    # add label to text classifier
    print("Add labels to text classifier")
    for label in data_headers:
        textcat.add_label(label)

    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    test_text = u"Notwithstanding anything to the contrary in this Agreement, each of the Indemnified Parties has " \
                u"relied on this Section 13.9, is an express third party beneficiary of this Section 13.9 and is " \
                u"entitled to enforce the obligations of the applicable Indemnified Parties under this Section 13.9 " \
                u"directly against such Indemnified Parties to the full extent thereof. "
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    # print(f"docs: {docs}")
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        # print(f"gold: {gold}")
        # print(f"Categoy items: {doc.cats.items()}")
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)

# BERT-based approaches:
# 1) https://keras.io/examples/nlp/semantic_similarity_with_bert/ - Seems like a very comphrensive approach
