#!/usr/bin/env python3
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URL)
db = client['app_database']

# Find the outfit
outfit = db['outfits'].find_one({'title': 'Athleisure Morning Run'})
if outfit:
    print(f'Current image URL: {outfit.get("image", "MISSING")}')
    
    # Update with a valid athleisure/running image
    new_image_url = "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=800&q=80"
    
    result = db['outfits'].update_one(
        {'title': 'Athleisure Morning Run'},
        {'$set': {'image': new_image_url}}
    )
    
    print(f'✓ Updated image URL to: {new_image_url}')
    print(f'  Modified count: {result.modified_count}')
    
    # Verify
    updated = db['outfits'].find_one({'title': 'Athleisure Morning Run'})
    print(f'✓ Verified new image URL: {updated.get("image")}')
else:
    print('❌ Outfit "Athleisure Morning Run" not found!')
