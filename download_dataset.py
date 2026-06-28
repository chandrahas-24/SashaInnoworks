import os
from icrawler.builtin import BingImageCrawler  # ← swap this

TARGET_INVENTORY = {
    "Carrot": "raw carrots",
    "Toaster": "kitchen toaster",
    "Book_pack": "stack of books",
    "Ceramic_mug": "coffee mug",
    "Cups": "plastic cups",
    "Cutlery": "cutlery set",
    "Glass_bottle": "glass beverage bottle",
    "Water_bottle": "plastic water bottle",
    "Almond_pouch": "almonds snack pouch",
    "Cloth_roll": "roll of cloth textile",
    "Detergent_pack": "laundry detergent packet",
    "Flour_sack": "sack of wheat flour",
    "Medicine_box": "medicine box packaging",
    "Oil_pouch": "cooking oil pouch",
    "Plastic_container": "clear plastic food container",
    "Rice_bag": "sack of raw rice",
    "Snack_packet": "chips packet",
    "Soap_bar_pack": "wrapped bar of soap",
    "Spice_pouch": "spice packet packaging",
    "Spices": "indian spices bowls",
    "Sugar_bag": "sugar paper bag",
    "Toy_box": "toy cardboard box retail",
    "Yogurt": "yogurt plastic container tub",
    "Blueberry": "fresh blueberries",
    "Cherry": "red cherries",
    "Custard_apple": "custard apple fruit",
    "Dragon_fruit": "dragon fruit",
    "Garlic": "garlic bulb",
    "Guava": "fresh green guava",
    "Kiwi": "kiwi fruit",
    "Lychee": "lychee fruit",
    "Mango": "ripe mango fruit",
    "Papaya": "papaya fruit",
    "Plum": "ripe purple plum",
    "Cereal": "cereal box",
    "Electronics_item": "small electronics router",
    "Grater": "metal cheese grater",
    "Kettle": "electric kettle",
}

DOWNLOAD_DIR = "raw_training_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
IMAGES_PER_CLASS = 80

print(f"Starting Bing crawl. Goal: {IMAGES_PER_CLASS} images per class.\n")

for cls, query in TARGET_INVENTORY.items():
    class_folder = os.path.join(DOWNLOAD_DIR, cls)
    os.makedirs(class_folder, exist_ok=True)

    # Check how many images already exist to skip completed classes
    existing = len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    if existing >= IMAGES_PER_CLASS:
        print(f"⏭️  Skipping [{cls}] — {existing} images already present")
        continue

    print(f"Scraping [{cls}] → '{query}' ({existing} already downloaded)")

    crawler = BingImageCrawler(
        storage={'root_dir': class_folder},
        log_level=40,
        downloader_threads=4,   # parallel downloads per class
    )

    crawler.crawl(
        keyword=query,
        filters={
            'type': 'photo',
            'size': 'large',
        },
        max_num=IMAGES_PER_CLASS,
        file_idx_offset='auto',
    )

    final_count = len([
        f for f in os.listdir(class_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    print(f"✅ [{cls}] done — {final_count} images in folder\n")

print("All classes complete.")