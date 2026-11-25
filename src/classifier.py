import requests
import time
import os
import re

# Hugging Face Inference Router URLs.
# Set HF_API_TOKEN in environment for authenticated access.
HF_API_URLS = {
    'animal': "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224",
    'cattle': "https://router.huggingface.co/hf-inference/models/microsoft/resnet-50"
}

# Breed mapping for better recognition - Updated with Indian breeds
CATTLE_BREEDS = {
    # International Dairy Breeds
    'holstein': 'Holstein Friesian (Dairy Cattle)',
    'jersey': 'Jersey (Dairy Cattle)',
    # International Beef Breeds
    'angus': 'Angus (Beef Cattle)',
    'hereford': 'Hereford (Beef Cattle)',
    'brahman': 'Brahman (Beef Cattle)',
    # Indian Dairy Breeds (High Milk Yield)
    'sahiwal': 'Sahiwal (Dairy Cattle - 2325 kg/lactation)',
    'gir': 'Gir (Dairy Cattle)',
    'red sindhi': 'Red Sindhi (Dairy Cattle - 1100-2600 kg/lactation)',
    'tharparkar': 'Tharparkar (Dairy Cattle - 1749 kg/lactation)',
    'rathi': 'Rathi (Dairy Cattle - 1560 kg/lactation)',
    # Indian Dual Purpose Breeds
    'kankrej': 'Kankrej (Dual Purpose)',
    'ongole': 'Ongole (Draught & Meat)',
    'hariana': 'Hariana (Dual Purpose)',
    'hallikar': 'Hallikar (Draught)',
    'kangayam': 'Kangayam (Draught)',
    'khillari': 'Khillari (Draught)',
    # Dwarf Breeds
    'vechur': 'Vechur (Dairy - Smallest Cattle Breed)',
    'punganur': 'Punganur (Dairy - Dwarf Breed)',
    # Generic Terms
    'ox': 'Cattle (Ox)',
    'bull': 'Cattle (Bull)',
    'cow': 'Cattle (Cow)',
    'cattle': 'Cattle',
    'calf': 'Cattle (Calf)'
}

BUFFALO_BREEDS = {
    # High-yielding River Buffalo Breeds
    'murrah': 'Murrah Buffalo (Delhi/Kundi - High Yielder)',
    'nili-ravi': 'Nili-Ravi Buffalo (Panch Kalyani)',
    'pandharpuri': 'Pandharpuri Buffalo (1790 kg/lactation)',
    'surti': 'Surti Buffalo (Charotari/Nadiadi - High Fat)',
    # Medium-yielding Breeds
    'bhadawari': 'Bhadawari Buffalo (Etawah - Dual Purpose)',
    'jaffrabadi': 'Jaffrabadi Buffalo',
    'mehsana': 'Mehsana Buffalo',
    # Regional Breeds
    'toda': 'Toda Buffalo (Nilgiri Hills - 8.22% Fat)',
    'chhattisgarhi': 'Chhattisgarhi Buffalo (Desi)',
    'gojri': 'Gojri Buffalo (Hill Breed)',
    'kalahandi': 'Kalahandi Buffalo',
    'chilika': 'Chilika Buffalo (Aquatic Vegetation)',
    # Generic Terms
    'buffalo': 'Water Buffalo',
    'water buffalo': 'Water Buffalo'
}

# Non-animal labels that can accidentally match terms like "cow" via substring checks.
NON_ANIMAL_LABEL_BLACKLIST = {
    'cowboy hat',
    'cowboy boot',
    'cowbell'
}

# Confidence thresholds for safe breed analysis.
MIN_LIVESTOCK_CONFIDENCE = 0.40
MIN_AI_BREED_CONFIDENCE = 0.55
MIN_SPECIES_SUPPORT_FOR_FALLBACK = 0.35


def _normalize_label(label):
    return re.sub(r'[^a-z0-9\s-]+', ' ', label.lower()).strip()


def _contains_term(label, term):
    """Return True when term appears as a whole word/phrase in label."""
    normalized_label = _normalize_label(label)
    normalized_term = _normalize_label(term)

    if not normalized_label or not normalized_term:
        return False

    escaped = re.escape(normalized_term).replace(r'\ ', r'\s+')
    pattern = rf'(^|\s){escaped}($|\s)'
    return re.search(pattern, normalized_label) is not None


def _is_valid_animal_label(label):
    normalized_label = _normalize_label(label)
    if normalized_label in NON_ANIMAL_LABEL_BLACKLIST:
        return False

    animal_terms = [
        'cattle', 'cow', 'bull', 'calf', 'ox', 'heifer',
        'buffalo', 'water buffalo', 'bison', 'bovine'
    ]
    return any(_contains_term(normalized_label, term) for term in animal_terms)


def _extract_best_breed_from_predictions(predictions):
    """
    Return best (animal_type, breed, confidence) from model predictions.
    Uses strict term matching to avoid false positives (e.g. "cowboy hat").
    """
    best_animal_type = "Animal"
    best_breed = "Unknown Breed"
    best_confidence = 0.0

    for pred in predictions[:10]:
        label = pred.get('label', '')
        score = pred.get('score', 0.0)

        if not _is_valid_animal_label(label):
            continue

        normalized_label = _normalize_label(label)

        for key, value in CATTLE_BREEDS.items():
            if _contains_term(normalized_label, key):
                if score > best_confidence:
                    best_animal_type = "Cattle"
                    best_breed = value
                    best_confidence = score
                break

        for key, value in BUFFALO_BREEDS.items():
            if _contains_term(normalized_label, key):
                if score > best_confidence:
                    best_animal_type = "Buffalo"
                    best_breed = value
                    best_confidence = score
                break

    return best_animal_type, best_breed, best_confidence


def _best_livestock_prediction(predictions):
    """
    Return best livestock-matching prediction from a model response.
    """
    best_label = "unknown"
    best_score = 0.0

    if not predictions or not isinstance(predictions, list):
        return best_label, best_score

    for pred in predictions[:10]:
        label = pred.get('label', '')
        score = float(pred.get('score', 0.0))
        if _is_valid_animal_label(label) and score > best_score:
            best_label = label
            best_score = score

    return best_label, best_score


def verify_livestock_image(image_path):
    """
    Verify whether the uploaded image likely contains cattle or buffalo.
    Returns: (is_livestock, confidence, reason)
    """
    result = query_huggingface_api(image_path, HF_API_URLS['animal'])

    if not result or not isinstance(result, list):
        return None, 0.0, "Unable to verify species from AI service. Try again or set HF_API_TOKEN for reliable analysis."

    best_label, best_score = _best_livestock_prediction(result)
    if best_score < MIN_LIVESTOCK_CONFIDENCE:
        top = result[0] if result else {}
        top_label = top.get('label', 'unknown')
        top_score = float(top.get('score', 0.0))
        return (
            False,
            best_score,
            f"No cattle or buffalo detected. Top prediction: {top_label} ({top_score*100:.1f}%)"
        )

    return True, best_score, f"Livestock detected: {best_label} ({best_score*100:.1f}%)"

def classify_animal(measurements):
    """
    Classify the animal based on measurements.
    This is a simple placeholder. In reality, this would use a trained ML model.
    """
    if not measurements:
        return 0
    
    # Simple scoring based on measurements
    score = 0
    if measurements.get('body_length', 0) > 100:
        score += 1
    if measurements.get('height_at_withers', 0) > 80:
        score += 1
    if measurements.get('chest_width', 0) > 50:
        score += 1
    if measurements.get('rump_angle', 0) < 50:
        score += 1
    
    return score

def query_huggingface_api(image_path, api_url, max_retries=3):
    """
    Query Hugging Face Inference API with image.
    Uses HF_API_TOKEN when available, but also attempts anonymous access.
    """
    token = os.getenv('HF_API_TOKEN')

    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            headers = {
                "Content-Type": "application/octet-stream"
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"

            response = requests.post(api_url, headers=headers, data=image_data, timeout=30)

            if response.status_code == 503:
                print(f"Model loading, waiting... (attempt {attempt+1}/{max_retries})")
                time.sleep(3)
                continue

            if response.status_code in (401, 403):
                if token:
                    print("HF API auth failed. Set a valid HF_API_TOKEN.")
                else:
                    print("HF API requires authentication. Set HF_API_TOKEN.")
                return None

            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                return None

            return response.json()
        except Exception as e:
            print(f"Error querying API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None

def classify_species(image_path):
    """
    Classify species using Hugging Face AI models.
    """
    try:
        result = query_huggingface_api(image_path, HF_API_URLS['animal'])
        
        if not result or not isinstance(result, list):
            return "unknown", 0.0
        
        # Get top predictions
        for pred in result[:5]:  # Check top 5 predictions
            label = pred.get('label', '')
            score = pred.get('score', 0.0)
            
            # Check for cattle/buffalo in label
            if any(_contains_term(label, term) for term in ['ox', 'cattle', 'cow', 'bull', 'calf']):
                return 'cattle', score
            elif any(_contains_term(label, term) for term in ['buffalo', 'water buffalo', 'bison']):
                return 'buffalo', score
        
        # Return top prediction if no specific match
        if result:
            return result[0].get('label', 'unknown'), result[0].get('score', 0.0)
        
        return "unknown", 0.0
    except Exception as e:
        print(f"Error in species classification: {e}")
        return "unknown", 0.0

def classify_breed_ai(image_path):
    """
    Classify breed using AI-based image recognition with Hugging Face models.
    """
    try:
        # First, get general classification
        result = query_huggingface_api(image_path, HF_API_URLS['cattle'])
        
        if not result or not isinstance(result, list):
            return "Animal", "Unknown Breed", 0.0
        
        # Analyze predictions with strict matching to avoid substring false positives.
        best_animal_type, best_breed, best_confidence = _extract_best_breed_from_predictions(result)
        
        # If still not found, use species classification
        if best_breed == "Unknown Breed":
            species, species_conf = classify_species(image_path)
            if species != "unknown":
                if 'buffalo' in species.lower():
                    best_animal_type = "Buffalo"
                    best_breed = "Water Buffalo"
                    best_confidence = species_conf
                elif any(term in species.lower() for term in ['cattle', 'ox', 'cow', 'bull']):
                    best_animal_type = "Cattle"
                    best_breed = "Domestic Cattle"
                    best_confidence = species_conf
        
        return best_animal_type, best_breed, best_confidence
    
    except Exception as e:
        print(f"Error in AI breed classification: {e}")
        return "Animal", "Unknown Breed", 0.0

def _estimate_breed_from_measurements(measurements):
    """
    Estimate specific breed from image-derived measurements.
    Uses multi-factor analysis of size, proportions, and coloration.
    """
    if not measurements:
        return "Animal", 0.3

    length = float(measurements.get('body_length', 0))
    height = float(measurements.get('height_at_withers', 0))
    chest = float(measurements.get('chest_width', 0))
    aspect_ratio = float(measurements.get('aspect_ratio', 0))
    compactness = float(measurements.get('compactness', 0))
    avg_brightness = float(measurements.get('avg_brightness', 0))

    # Calculate derived features
    size_score = length * height
    body_mass_index = compactness / max(size_score, 1) * 1000

    breed_scores = {}

    # BUFFALO BREEDS
    # Murrah Buffalo - Large, dark, stocky build
    if avg_brightness < 130 and length > 400 and height > 350:
        breed_scores["Murrah Buffalo"] = 0.65
    elif avg_brightness < 130 and aspect_ratio < 1.2 and size_score > 150000:
        breed_scores["Murrah Buffalo"] = 0.55

    # Nili-Ravi Buffalo - Medium-large, moderate brightness
    if 130 <= avg_brightness < 160 and 1.3 <= aspect_ratio <= 2.0 and length > 350:
        breed_scores["Nili-Ravi Buffalo"] = 0.60
    elif compactness > 1800 and 400 <= length <= 550:
        breed_scores["Nili-Ravi Buffalo"] = 0.50

    # Bhadawari Buffalo - Smaller, compact, darker
    if avg_brightness < 120 and length < 300 and compactness < 800:
        breed_scores["Bhadawari Buffalo"] = 0.60
    elif size_score < 40000 and avg_brightness < 130:
        breed_scores["Bhadawari Buffalo"] = 0.50

    # Jaffrabadi Buffalo - Very large and heavy
    if length > 500 and height > 450 and avg_brightness < 125:
        breed_scores["Jaffrabadi Buffalo"] = 0.55

    # CATTLE BREEDS
    # Jersey - Medium size, higher brightness, good proportions
    if 130 <= avg_brightness <= 150 and 1.3 <= aspect_ratio <= 1.7:
        if 800 <= length <= 1000 and 400 <= height <= 550:
            breed_scores["Jersey (Dairy Cattle)"] = 0.65
        elif chest > 400 and size_score > 350000:
            breed_scores["Jersey (Dairy Cattle)"] = 0.55

    # Holstein Friesian - Large, tall, bright
    if avg_brightness > 140 and length > 900 and height > 500:
        breed_scores["Holstein Friesian (Dairy Cattle)"] = 0.65
    elif size_score > 450000 and aspect_ratio < 2.0:
        breed_scores["Holstein Friesian (Dairy Cattle)"] = 0.55

    # Sahiwal - Medium size, moderate proportions
    if 140 <= avg_brightness <= 160 and 200 <= length <= 350:
        if 1.5 <= aspect_ratio <= 2.0:
            breed_scores["Sahiwal (Dairy Cattle)"] = 0.60
        elif compactness < 500:
            breed_scores["Sahiwal (Dairy Cattle)"] = 0.50

    # Gir - Distinctive proportions, moderate size
    if 300 <= length <= 500 and 1.2 <= aspect_ratio <= 1.6:
        if avg_brightness > 135:
            breed_scores["Gir (Dairy Cattle)"] = 0.50

    # General species fallback based on characteristics
    if not breed_scores:
        # Handle extreme aspect ratios (likely measurement artifacts)
        if aspect_ratio > 4.0:
            # Use other features
            if avg_brightness < 135 and compactness > 800:
                return "Water Buffalo", 0.35
            elif avg_brightness > 135:
                return "Domestic Cattle", 0.35
            
        # Dark animals are more likely buffalo
        if avg_brightness < 128:
            if aspect_ratio < 1.2 or compactness > 1500:
                return "Water Buffalo", 0.40
            return "Buffalo", 0.35
        # Bright animals with good proportions are likely cattle
        elif avg_brightness > 140:
            if 1.3 <= aspect_ratio <= 1.8:
                return "Domestic Cattle", 0.40
            return "Cattle", 0.35
        # Mid-range - analyze proportions
        elif compactness > 1500 or aspect_ratio < 1.1:
            return "Buffalo", 0.30
        elif 1.2 <= aspect_ratio <= 2.0:
            return "Cattle", 0.30
        return "Animal", 0.25

    # Return highest scoring breed
    best_breed = max(breed_scores.items(), key=lambda x: x[1])
    return best_breed[0], best_breed[1]


def classify_breed(measurements, image_path):
    """
    Classify the breed using AI-based image recognition.
    Falls back to measurement-based breed estimation if AI is unavailable.
    """
    if not image_path:
        return "Unknown - No image provided"

    is_livestock, livestock_confidence, verification_message = verify_livestock_image(image_path)
    verification_unavailable = is_livestock is None

    if is_livestock is False:
        return verification_message

    animal_type, breed, confidence = classify_breed_ai(image_path)

    # Use AI prediction if confidence is high enough
    if confidence > MIN_AI_BREED_CONFIDENCE:
        if confidence > 0.7:
            return f"{breed} (AI: {confidence*100:.1f}% confident) ✓"
        return f"{breed} (AI: {confidence*100:.1f}% confident)"

    # Fallback to measurement-based estimation
    if not measurements:
        if verification_unavailable:
            return "AI service unavailable and image measurements are insufficient for breed analysis."
        return "Low-confidence analysis. Upload a clearer side-view cattle/buffalo image."

    # If species verification is available, only allow fallback when evidence is reasonably strong.
    if (not verification_unavailable) and livestock_confidence < MIN_SPECIES_SUPPORT_FOR_FALLBACK:
        return "Low-confidence analysis. Could not confirm breed reliably from this image."

    estimated_breed, est_confidence = _estimate_breed_from_measurements(measurements)

    if verification_unavailable:
        # AI service may be rate-limited/unavailable. Provide clear offline result wording.
        if est_confidence >= 0.55:
            return f"{estimated_breed} (Estimated offline: {est_confidence*100:.0f}% - AI service unavailable)"
        return "AI service unavailable. Measurement-only estimate is not reliable for this image."
    
    # If AI gave us species info but low confidence, prefer AI species with measurement breed
    if animal_type in ("Buffalo", "Cattle") and confidence > 0.3:
        if animal_type == "Buffalo" and "Buffalo" not in estimated_breed:
            estimated_breed = "Water Buffalo"
            est_confidence = max(est_confidence, confidence)
        elif animal_type == "Cattle" and "Cattle" not in estimated_breed and "Buffalo" in estimated_breed:
            estimated_breed = "Domestic Cattle"
            est_confidence = max(est_confidence, confidence)

    if est_confidence >= 0.50:
        return f"{estimated_breed} (Estimated: {est_confidence*100:.0f}%)"
    else:
        return f"{estimated_breed} (Estimated from measurements)"
