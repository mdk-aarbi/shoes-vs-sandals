SHOES vs SANDALS CLASSIFIER
Deep Learning semester project, BSCS, Superior University, Lahore
Author: Muhammad Danial Khalid
Course teacher: Dr. Munir Ahmad

Live website:      https://shoes-vs-sandals.vercel.app
Full dataset:      https://drive.google.com/file/d/1A2PE72WpTZ90VgjiFMs4JroN4JwNBwp7/view?usp=sharing
Project report:    https://claude.ai/artifact/UEcmQ2oZbqGacbZrug6JFa
Presentation:      https://claude.ai/artifact/D1Sd7vyEeX8FziUcPZ2KxY
YouTube video:     <PASTE LINK HERE>


1. WHAT THIS PROJECT DOES
-------------------------
Given a photo of footwear, the models say whether it is a SHOE (closed shoe: sneakers,
joggers, formal shoes, boots, loafers) or a SANDAL (any open style: sandals, chappals,
Peshawari chappal, slippers, flip-flops).

Two models were trained on the same self-collected dataset:
  a) Self-built CNN, trained from random weights (self learning)
  b) MobileNetV2 pretrained on ImageNet, fine-tuned (transfer learning)


2. FOLDER MAP
-------------
index.html                  The website (deployed on Vercel). Runs both models in the browser.
assets/tf.min.js            TensorFlow.js 4.22.0, used by the website.
web_models/                 Both models converted to TensorFlow.js format for the website.
models/                     Final trained weights of both models:
  self_cnn.weights.h5                       Self-built CNN weights
  transfer_mobilenetv2.weights.h5.part00-02 Transfer-learning weights, split into 3 parts
                                            (GitHub's upload page limits file size;
                                            predict.py joins them automatically)
  classes.json                              Label order and preprocessing rules
                            The architectures are defined in predict.py (same as the notebook).
                            Full saved models (.keras) backup: https://drive.google.com/file/d/1jgtxzA_jALeDtcvq7YvK5q09VirZUDNk/view?usp=sharing
predict.py                  Command-line prediction for one photo or a whole folder.
requirements.txt            Python packages for predict.py.
test_samples/               39 held-out single photos (shoes/ and sandals/) for quick testing.
results/                    Training curves, confusion matrices, augmentation examples, results.json.
notebook/shoes_vs_sandals_training.ipynb
                            Complete training notebook (Google Colab): data preparation,
                            both models, evaluation, saving weights, TensorFlow.js export.
scripts/extract_frames.py   Turns the class videos into labelled frames with a train/test split.


3. HOW TO RUN THE MODELS
------------------------
    pip install -r requirements.txt

    python predict.py test_samples/shoes/shoes_01.jpg                 (transfer model)
    python predict.py test_samples/shoes/shoes_01.jpg --model cnn     (self-built CNN)
    python predict.py test_samples --model transfer                   (whole folder + accuracy)

Input preprocessing (same in training, predict.py and the website):
fix EXIF rotation, convert to RGB, resize to 224x224, scale pixels with x / 127.5 - 1.
Output: one number between 0 and 1 = probability the photo is a shoe (>= 0.5 means shoe).

To load in your own code (builds the architecture, then loads the trained weights):
    from predict import load_model
    model = load_model("transfer")      # or load_model("cnn")


4. DATASET
----------
Primary dataset collected for this project (not downloaded from Kaggle).
  - Videos recorded at shoe shops and racks, one class per video. Frames sampled every
    0.5 s, blurry and near-duplicate frames removed (scripts/extract_frames.py).
  - Single photos contributed by family, friends and colleagues through a consent form.
  - Faces and personal details excluded; phone metadata (location, device, time) stripped;
    file names anonymised.

Split                   Shoes   Sandals   Notes
Train                   1,175   1,009     frames from 31 + 41 videos, plus 53 + 44 single photos
Test (video frames)        13      12     one whole video per class, never seen in training
Test (single photos)       20      20     pairs not in training

Whole videos were kept in train OR test, never both, so the test frames are not near-copies
of training frames. The full dataset is at the Google Drive link at the top of this file.


5. TRAINING
-----------
Preprocessing: resize 224x224, scale to -1..1. Training-only augmentation: horizontal flip,
rotation, zoom, contrast, brightness. Single photos weighted 5x a video frame; classes balanced.

Self-built CNN: 4 blocks of Conv2D-BatchNorm-ReLU-MaxPool (32, 64, 128, 128 filters),
global average pooling, dropout 0.4, sigmoid output. 40 epochs, Adam, cosine learning-rate
decay from 1e-3, milder augmentation.

Transfer learning: MobileNetV2 (ImageNet weights) + dropout 0.3 + sigmoid output.
Stage 1: base frozen, head trained 6 epochs at 1e-3. Stage 2: last 30 layers unfrozen,
6 epochs at 1e-5.

Loss: binary cross-entropy. Hardware: Google Colab, T4 GPU.


6. RESULTS
----------
Model                       Video frames (n=25)    Single photos (n=40)
Transfer learning           25/25 (100%)           29/40 (72.5%)
Self-built CNN              17/25 (68%)            19/40 (47.5%)

The single-photo test is the honest measure, because the website receives single photos.
The test sets are small: one image moves accuracy by 2.5 to 4 points, and repeated training
runs vary by a few points.

Findings:
  - Transfer learning clearly beats the CNN trained from scratch on a small dataset.
  - The self-built CNN is near chance on single photos: it learned rack and shelf
    appearance rather than footwear features.
  - Most training data is rack footage; more single-item photos is the main improvement.

test_samples/ holds 39 of the 40 single test photos (one internet stock image was removed
before publishing), so predict.py on that folder gives a slightly different figure.


7. WEBSITE (VERCEL)
-------------------
Static site, no server code. Vercel serves index.html; the browser loads web_models/ with
TensorFlow.js and classifies the uploaded photo locally. .vercelignore keeps the large
.h5 weight files and the notebook out of the deployment (they stay on GitHub).


8. AI USE DISCLOSURE
--------------------
AI assistants (Claude) were used to help write code, the website and documentation, as
permitted for this project. Data collection, training runs and evaluation were done by the
author, and the author can explain every part of the code.
