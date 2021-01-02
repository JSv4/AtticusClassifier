import zipfile
import pickle
import spacy
import io
from lexnlp.nlp.en.segments.sentences import pre_process_document, normalize_text, get_sentence_list

pickle_zip = zipfile.ZipFile('./pre-trained/BertClassifierPickle.zip')
pickle_bytes = io.BytesIO(pickle_zip.read('BertClassifier.pickle'))

nlp_pickle = pickle.load(pickle_bytes)

text="""JOINT VENTURE AGREEMENT
Collectible Concepts Group, Inc. ("CCGI") and Pivotal Self Service Tech, Inc.
("PVSS"), (the "Parties" or "Joint Venturers" if referred to collectively, or
the "Party" or Joint Venturer" if referred to singularly), by this Agreement
associate themselves as business associates, and not as partners, in the
formation of a joint venture (the "Joint Venture"), for the purpose of engaging
generally in the business provided for by terms and provisions of this
Agreement.
1. Name of the Joint Venture. The name of the Joint Venture will be MightyCell
Batteries, and may sometimes be referred to as "MightyCell" or the "Joint
Venture" in this Agreement. The principal office and place of business
shall be located in 1600 Lower State Road, Doylestown, PA 18901.
2. Scope of the Joint Venture Business. The Joint Venture is formed for the
purpose of engaging generally in the business of marketing batteries and
related products, (the "Products") that include the display of licensed
logos, images, brand names and other labels that differentiate them from
the branding (the "PVSS Products") under which PVSS and/or its affiliates,
sell to retailers and distributors in the normal course of their business.
Without in any way limiting the generality of the foregoing, the business
of the Joint Venture shall include:
The purchase of Products for resale;
The acquisition of a license(s) permitting the use of selected
images in the Products;
The sale and distribution of the Products to retailers and
distributors; and,
The transaction of such other and further business as is
necessary, advisable, or incidental to the business of the Joint
Venture.
Develop a global marketing program for licensed Products
Exhibit A attached hereto, describes by way of example but not limitation
the responsibilities of the Joint Venturers
3. Capital Contributions. Except as agreed upon by mutual consent, the Joint
Venturers shall not be required to make any capital contribution to the
Joint Venture.
4. Offices of the Joint Venture. The principal place of business of the Joint
Venture shall be at 1600 Lower State Road, in the City of Doylestown, Bucks
County, Pennsylvania, but may maintain such other offices as the Joint
Venturers may deem advisable at any other place or places within or without
the Commonwealth of Pennsylvania.
5. Powers and Authority of the Joint Venturers. The Joint Venturers shall have
full and complete charge of all affairs of the Joint Venture. The Joint
Venturers recognize that both of the Joint Venturers are and will continue
to be engaged in the conduct of their respective businesses for their own
account. Neither Joint Venturer shall be entitled to compensation for
services rendered to the Joint Venture as such, but each Joint Venturer
shall be reimbursed for all direct expenses, including travel, office, and
all other out-of-pocket expenses incurred in the operation of the affairs
of the Joint Venture and the promotion of its businesses.
It is agreed that either Joint Venturer shall, except as provided for
below, have authority to execute instruments of any character relating to
the affairs of the Joint Venture; provided, that without the written
consent or approval of both of the Joint Venturers: (i) the Joint Venture
shall incur no liability of any sort, nor any indebtedness for borrowed
funds; (ii) no assets owned in the name of the Joint Venture be disposed
of; and (iii) no commitment to purchase any item for the Joint Venture
shall be made.
Division of Income and Losses. All income and credits, and all losses and
deductions shall be owned and shared among the Joint Venturers as follows:
50% to Collectible Concepts Group, Inc.
50% to Pivotal Self Service Tech, Inc.
Depreciation and all other charges and expenses, which are not expressly
apportioned by this Agreement, shall be apportioned in accordance with
generally accepted accounting principles, consistently applied.
Accounting Provisions. The Joint Venturers shall maintain adequate books
and records to be kept of all the Joint Venture activities and affairs
conducted pursuant to the terms of this Agreement. All direct costs and
expenses, which shall include any insurance costs in connection with the
distribution of the Products or operations of the Joint Venture, or if the
business of the Joint Venture requires additional office facilities than
those now presently maintained by each Joint Venturer"""


print("#############################################\nTest Pickled Model\n")
for sent in get_sentence_list(normalize_text(pre_process_document(text))):
    print(f"LexNLP Sentence: {sent}")
    cats = nlp_pickle(sent).cats
    cats = [label for label in cats if cats[label] > .7]
    print(cats)
