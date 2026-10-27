# ShipSmart - AI Food Item Classifier (START HERE)

You only do TWO things yourself: add your photos, then run a few commands.
Then scan items with your phone using Expo Go.

## What's inside
```
SmartPantry_Complete/
├── dataset/              <- PUT YOUR PHOTOS HERE (one folder per food)
│   ├── apple/  banana/  bread/  egg/
│   └── milk_carton/  tomato/  packaged_snack/  canned_food/
├── split_dataset.py      step 1: splits photos into train/val/test
├── train_model.py        step 2: trains YOUR model
├── app.py + templates/   step 3: server (also a web page) using your model
├── expo_app/             step 4: phone app for Expo Go
├── extract_frames.py     (optional) turns a video into photos
└── requirements.txt
```

## Step 0 - Install (once)
Python 3.10 or 3.11, then in a terminal in this folder:
```
pip install -r requirements.txt
```
The app also loads an open-vocabulary CLIP model the first time it starts.
This model can recognize common foods that are not in your training folders,
including onions and potatoes. An internet connection is needed on first start
to download its weights. Its candidate labels cover common grocery items, but
predictions for unfamiliar foods are not guaranteed.

## Step 1 - Add your photos
Drag photos into the matching folders in `dataset/`.
- 100+ per folder is ideal (30 is enough for a test); vary angle, light, background
- You never paste images into code - the scripts read the folders
- Different foods? Rename/add/delete folders. Folder names become labels.
  Keep at least 2 folders with photos and delete any empty ones.
- Shortcut: film a 30 sec video, then `python extract_frames.py video.mp4 apple`

## Step 2 - Train
```
python split_dataset.py
python train_model.py
```
Accuracy numbers for your report: `model/results_summary.txt`, `training_history.png`.

## Step 3 - Start the server
```
python app.py
```
It prints the address to use on your phone. (You can also open
http://127.0.0.1:5000 on the computer. Green banner = your own model.)

## Step 4 - Use your phone with Expo Go
Follow `expo_app/README.md`.

## Host the website on GitHub Pages

GitHub Pages serves the static browser interface from `docs/`. Predictions are
handled separately by the Flask/TensorFlow/CLIP server on Render; GitHub Pages
cannot run Python or the classifier itself.

1. Create a GitHub repository and upload this project's files, including
   `.github/workflows/pages.yml`, `docs/`, `Dockerfile`, `render.yaml`,
   `requirements.txt`, `app.py`, and `model/food_classifier.h5` plus
   `model/class_names.json`. Do not upload `.venv/`, `dataset/`, or
   `dataset_split/`.
2. Push the files to the repository's `main` or `master` branch. In the
   repository's **Settings → Pages**, set the build and deployment source to
   **GitHub Actions**. The workflow deploys `docs/` and publishes the Pages URL.
3. In Render, create a Blueprint from that repository and deploy the `shipsmart`
   web service defined in `render.yaml`. The Standard instance is selected
   because the model server used about 1.1 GB of memory on this computer.
4. The static page defaults to `https://shipsmart.onrender.com` for its
   prediction API. If Render assigns a different service URL, update
   `API_BASE_URL` in `docs/index.html` to that URL and push the change.
5. Wait for the Render service's first deploy to download the CLIP model. Open
   the GitHub Pages URL over HTTPS in Chrome; the site will send photos to the
   prediction server and can request camera permission.

On GitHub Free, Pages generally requires a public repository, which makes the
repository's source files publicly viewable. If you need private source, use a
GitHub plan that supports Pages from private repositories. In either case,
visitors to the website do not get access to files on your computer. The hosting
provider will receive the model and application files needed to run predictions.

## Troubleshooting
- "No images found yet": put photos in the `dataset/` folders first.
- Training errors about empty classes: delete empty folders in `dataset/`.
- Training slow: use Google Colab or fewer images first.
- After retraining, restart `app.py`.
