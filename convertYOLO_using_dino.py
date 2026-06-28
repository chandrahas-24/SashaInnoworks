import os
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

BASE_DIR = "/Users/chandrahass/PycharmProjects/SashaInnoworks/raw_training_data"
OUT_DIR  = "/Users/chandrahass/PycharmProjects/SashaInnoworks/yolo_dataset_dino"

TARGET_INVENTORY = {
    "raw carrot": "Carrot",
    "kitchen toaster": "Toaster",
    "stack of books": "Book_pack",
    "coffee mug": "Ceramic_mug",
    "plastic cup": "Cups",
    "cutlery set": "Cutlery",
    "glass beverage bottle": "Glass_bottle",
    "plastic water bottle": "Water_bottle",
    "almonds snack pouch": "Almond_pouch",
    "roll of cloth": "Cloth_roll",
    "laundry detergent packet": "Detergent_pack",
    "sack of flour": "Flour_sack",
    "medicine box": "Medicine_box",
    "cooking oil pouch": "Oil_pouch",
    "clear plastic food container": "Plastic_container",
    "rice bag": "Rice_bag",
    "chips snack packet": "Snack_packet",
    "wrapped soap bar": "Soap_bar_pack",
    "spice packet": "Spice_pouch",
    "indian spices in bowl": "Spices",
    "sugar bag": "Sugar_bag",
    "toy box": "Toy_box",
    "yogurt tub": "Yogurt",
    "fresh blueberries": "Blueberry",
    "red cherries": "Cherry",
    "custard apple fruit": "Custard_apple",
    "dragon fruit": "Dragon_fruit",
    "garlic bulb": "Garlic",
    "fresh guava": "Guava",
    "kiwi fruit": "Kiwi",
    "lychee fruit": "Lychee",
    "ripe mango": "Mango",
    "papaya fruit": "Papaya",
    "purple plum": "Plum",
    "cereal box": "Cereal",
    "electronic router": "Electronics_item",
    "metal grater": "Grater",
    "electric kettle": "Kettle",
}

for prompt, cls in TARGET_INVENTORY.items():
    input_folder = os.path.join(BASE_DIR, cls)
    output_folder = os.path.join(OUT_DIR, cls)

    if not os.path.exists(input_folder):
        print(f"⚠️  Missing folder: {cls}, skipping")
        continue

    img_count = len([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    if img_count == 0:
        print(f"⚠️  No images in {cls}, skipping")
        continue

    print(f"\n🔍 Labeling [{cls}] — {img_count} images — prompt: '{prompt}'")

    ontology = CaptionOntology({prompt: cls})
    model = GroundingDINO(ontology=ontology)
    model.label(
        input_folder=input_folder,
        output_folder=output_folder,
        extension=".jpg",
    )
    print(f"✅ [{cls}] done")

print("\n🎉 All classes done!")