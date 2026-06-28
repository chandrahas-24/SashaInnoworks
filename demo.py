import os
import requests
import random
import time
import torch
from ultralytics import YOLO

# Bulletproof DuckDuckGo import based on your earlier setup
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        print("Missing library! Run: pip install duckduckgo-search")
        exit()

# We map your exact classes to specific search queries to get good results
SEARCH_QUERIES = {
    "Grape": "bunch of fresh grapes on table",
    "Pomegranate": "pomegranate fruit open",
    "Strawberry": "fresh strawberries close up",
    "Watermelon": "watermelon whole and sliced",
    "Pear": "fresh pears organic",
    "Pineapple": "pineapple fruit whole",
    "Apple": "apples grocery store",
    "Banana": "bunch of bananas raw",
    "Orange": "fresh oranges fruit",
    "Coconut": "coconut whole and half",
    "Blender": "kitchen blender appliance",
    "Bread": "loaf of sliced bread bakery",
    "Cheese": "block of cheddar cheese",
    "Box": "cardboard boxes stacked warehouse",
    "Tin can": "tin cans pantry shelf"
}

IMAGES_PER_CLASS = 30
DEMO_FOLDER = "presentation_demo_images"


def download_image(url, save_path):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False


def build_demo_dataset():
    os.makedirs(DEMO_FOLDER, exist_ok=True)
    images_to_test = []

    print(f"--- 1. Scraping the Live Web for {len(SEARCH_QUERIES) * IMAGES_PER_CLASS} Images ---")

    with DDGS() as ddgs_client:
        for cls, query in SEARCH_QUERIES.items():
            print(f"\nPulling 30 images for: {cls} -> '{query}'")

            # Requesting extra just in case some links are dead
            results = list(ddgs_client.images(query, max_results=50))

            downloaded_count = 0
            for i, res in enumerate(results):
                if downloaded_count >= IMAGES_PER_CLASS:
                    break

                img_url = res.get('image')
                save_path = os.path.join(DEMO_FOLDER, f"web_{cls}_{i}.jpg")

                if download_image(img_url, save_path):
                    images_to_test.append(save_path)
                    downloaded_count += 1
                    # Tiny delay to prevent getting blocked by the search engine
                    time.sleep(0.1)

            print(f"✅ Secured {downloaded_count}/{IMAGES_PER_CLASS} for {cls}")

    # SHUFFLE THE ENTIRE DECK
    print("\n--- 2. Shuffling the Dataset ---")
    random.shuffle(images_to_test)
    print("Dataset completely randomized. Ready for inference.")

    return images_to_test


def run_inference(image_paths):
    if not image_paths:
        print("No images found to test!")
        return

    weights_path = "runs/detect/PIEDS_Vision/pieds_15_class-4/weights/best.pt"

    print(f"\n--- 3. Running YOLO Vision with Custom Weights ---")
    print(f"Loading weights from: {weights_path}")

    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nAnalyzing 450 shuffled images ONE BY ONE to bypass Apple memory limits... 🚀")

    # Manually loop through the files so PyTorch cannot pre-allocate memory
    for i, img_path in enumerate(image_paths):
        model.predict(
            source=img_path,
            save=True,
            conf=0.30,
            device="mps",
            verbose=False  # Keeps your terminal from spamming 450 times
        )

        # Manually flush the Mac's unified memory cache every 10 images
        if (i + 1) % 10 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"Processed {i + 1}/{len(image_paths)} images safely...")

    print("\n🎉 DEMO READY!")
    print("Open your PyCharm sidebar and look inside the newest 'runs/detect/predict...' folder.")
    print("Your full shuffled gallery is safe and fully annotated.")


if __name__ == "__main__":
    # If you already downloaded them on a previous run, skip straight to testing!
    if os.path.exists(DEMO_FOLDER) and len(os.listdir(DEMO_FOLDER)) > 100:
        print("✅ Found existing demo folder. Skipping download and jumping straight to inference!")
        existing_images = [os.path.join(DEMO_FOLDER, f) for f in os.listdir(DEMO_FOLDER) if f.endswith('.jpg')]
        random.shuffle(existing_images)
        run_inference(existing_images)
    else:
        fresh_images = build_demo_dataset()
        run_inference(fresh_images)