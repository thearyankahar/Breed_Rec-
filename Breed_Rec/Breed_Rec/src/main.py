from flask import Flask, request, jsonify, render_template
import os
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from urllib.request import urlopen
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_processor import process_image
from src.classifier import classify_animal, classify_breed
import logging

APP_BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(APP_BASE_DIR / 'templates'))

UPLOAD_FOLDER = APP_BASE_DIR / 'data' / 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_BREEDING_GOALS = {'balanced', 'dairy', 'draught', 'dual'}


TRANSLATION_OVERRIDES = {
    'gu': {
        'Computer Vision For Livestock': 'પશુધન માટે કમ્પ્યુટર વિઝન',
        'Image-Based Animal Type Classification for Cattle and Buffaloes': 'ઢોર અને ભેંસ માટે છબી-આધારિત પ્રાણી પ્રકારનું વર્ગીકરણ',
        'Unlocking precision in animal classification—fast, reliable, and designed for the future of smart farming.': 'પ્રાણી વર્ગીકરણમાં ચોકસાઈ મેળવવી - ઝડપી, વિશ્વસનીય અને સ્માર્ટ ખેતીના ભવિષ્ય માટે રચાયેલ.',
        'Fast visual scoring in a single run': 'એક જ રનમાં ઝડપી દૃશ્ય સ્કોરિંગ',
        'Structured metrics for reporting': 'રિપોર્ટિંગ માટે ગોઠવાયેલા મેટ્રિક્સ',
        '3D micro-interactions and smooth motion': '3D સૂક્ષ્મ ક્રિયાઓ અને સરળ ગતિ',
        'Upload And Analyze': 'અપલોડ કરો અને વિશ્લેષણ કરો',
        'For best output, upload a bright side-profile image with the animal body fully visible and minimal background clutter.': 'સૌથી સારું પરિણામ માટે, પ્રાણીનું શરીર સંપૂર્ણ દેખાતું હોય એવી તેજસ્વી બાજુ-પ્રોફાઇલ છબી અપલોડ કરો અને પૃષ્ઠભૂમિ ઓછી રાખો.',
        'Select animal image': 'પ્રાણીની છબી પસંદ કરો',
        'Breeding Goal Preference': 'સંવર્ધન ધ્યેય પસંદગી',
        'Image Preview': 'છબી પૂર્વદર્શન',
        'Preview will appear here after selecting a file': 'ફાઇલ પસંદ કર્યા પછી પૂર્વદર્શન અહીં દેખાશે',
        'Run Analysis': 'વિશ્લેષણ ચલાવો',
        'Supported: JPG, JPEG, PNG, WEBP': 'સમર્થિત: JPG, JPEG, PNG, WEBP',
        'Pipeline': 'પ્રક્રિયા ક્રમ',
        'Output': 'આઉટપુટ',
        'Analysis Results': 'વિશ્લેષણ પરિણામો'
    }
}


BREED_BASIC_INFO = {
    'murrah': {
        'animal_type': 'Buffalo',
        'origin': 'Haryana and Delhi',
        'primary_use': 'High-yield dairy buffalo',
        'highlights': 'Jet-black coat, tightly curled horns, strong milk production'
    },
    'nili-ravi': {
        'animal_type': 'Buffalo',
        'origin': 'Punjab belt (Sutlej-Ravi region)',
        'primary_use': 'Dairy',
        'highlights': 'Black body with white markings, known for quality milk'
    },
    'surti': {
        'animal_type': 'Buffalo',
        'origin': 'Gujarat',
        'primary_use': 'Dairy',
        'highlights': 'Compact body, sickle-shaped horns, high fat milk'
    },
    'pandharpuri': {
        'animal_type': 'Buffalo',
        'origin': 'Maharashtra',
        'primary_use': 'Dairy',
        'highlights': 'Very long horns, good lactation performance'
    },
    'bhadawari': {
        'animal_type': 'Buffalo',
        'origin': 'Uttar Pradesh and Madhya Pradesh',
        'primary_use': 'Dual purpose',
        'highlights': 'Copper-black shade, adaptable to rough fodder'
    },
    'sahiwal': {
        'animal_type': 'Cattle',
        'origin': 'Punjab and Rajasthan',
        'primary_use': 'Dairy cattle',
        'highlights': 'Heat-tolerant Indian dairy breed with strong milk yield'
    },
    'gir': {
        'animal_type': 'Cattle',
        'origin': 'Gujarat',
        'primary_use': 'Dairy cattle',
        'highlights': 'Hardy indigenous breed with good heat tolerance'
    },
    'red sindhi': {
        'animal_type': 'Cattle',
        'origin': 'Sindh line, widespread in India',
        'primary_use': 'Dairy cattle',
        'highlights': 'Reddish coat, heat-tolerant, reliable milk producer'
    },
    'tharparkar': {
        'animal_type': 'Cattle',
        'origin': 'Rajasthan and Gujarat',
        'primary_use': 'Dairy and dual purpose',
        'highlights': 'Drought-adapted, light grey-white coat, useful in arid zones'
    },
    'rathi': {
        'animal_type': 'Cattle',
        'origin': 'Rajasthan',
        'primary_use': 'Dairy cattle',
        'highlights': 'Desert-adapted, good milker with patchy coat patterns'
    },
    'holstein': {
        'animal_type': 'Cattle',
        'origin': 'Netherlands and Germany line',
        'primary_use': 'High-yield dairy cattle',
        'highlights': 'Large frame, black-white pattern, very high milk output'
    },
    'jersey': {
        'animal_type': 'Cattle',
        'origin': 'Jersey Island',
        'primary_use': 'Dairy cattle',
        'highlights': 'Smaller body, efficient feed use, high-fat milk'
    },
    'ongole': {
        'animal_type': 'Cattle',
        'origin': 'Andhra Pradesh',
        'primary_use': 'Draught and beef line',
        'highlights': 'Large and powerful indigenous breed'
    },
    'vechur': {
        'animal_type': 'Cattle',
        'origin': 'Kerala',
        'primary_use': 'Small-scale dairy',
        'highlights': 'One of the smallest cattle breeds in the world'
    },
}


BREED_FIELD_CONTEXT = {
    'murrah': {
        'climate_fit': 'Performs well in warm plains with reliable water access.',
        'management_tip': 'Best suited to dairy-oriented herds with strong feeding support.'
    },
    'nili-ravi': {
        'climate_fit': 'Fits irrigated and moderate-heat regions with regular stall care.',
        'management_tip': 'Useful when milk quality and steady lactation are priorities.'
    },
    'surti': {
        'climate_fit': 'Comfortable in warm western regions and semi-intensive systems.',
        'management_tip': 'Good for dairy lines where higher milk fat matters.'
    },
    'pandharpuri': {
        'climate_fit': 'Well aligned with hot, dry conditions when hydration is managed.',
        'management_tip': 'Useful for dairy-focused selection in hardy buffalo herds.'
    },
    'bhadawari': {
        'climate_fit': 'Works in hot regions and rougher feeding environments.',
        'management_tip': 'Suitable for mixed milk-and-work expectations under practical farm conditions.'
    },
    'sahiwal': {
        'climate_fit': 'Strong fit for hot climates and tropical dairy systems.',
        'management_tip': 'A practical option for heat-tolerant milk production.'
    },
    'gir': {
        'climate_fit': 'Performs well in warm climates and drought-prone regions.',
        'management_tip': 'Useful when hardiness and dairy value both matter.'
    },
    'red sindhi': {
        'climate_fit': 'Adapted to heat-heavy regions with moderate fodder support.',
        'management_tip': 'A solid dairy-line option for resilience-focused herds.'
    },
    'tharparkar': {
        'climate_fit': 'Well suited to arid and semi-arid conditions.',
        'management_tip': 'Helpful for farms balancing dairy value and field endurance.'
    },
    'rathi': {
        'climate_fit': 'Works well in dry inland zones and heat-stressed environments.',
        'management_tip': 'Fits dairy-oriented indigenous breeding programs.'
    },
    'holstein': {
        'climate_fit': 'Best in cooler or well-managed dairy environments with strong nutrition.',
        'management_tip': 'High output potential, but management quality must stay high.'
    },
    'jersey': {
        'climate_fit': 'More flexible across climates than larger exotic dairy breeds.',
        'management_tip': 'Useful for efficient dairy systems with limited feed budgets.'
    },
    'ongole': {
        'climate_fit': 'A strong fit for hot conditions and field-work environments.',
        'management_tip': 'Better aligned with rugged use and structural strength than milk-first breeding.'
    },
    'vechur': {
        'climate_fit': 'Fits humid smallholder systems and low-space management.',
        'management_tip': 'Useful for compact dairy setups and low-input conditions.'
    }
}


def _extract_core_breed_name(breed_prediction):
    """Strip confidence suffixes from classifier output while keeping breed label."""
    if not breed_prediction:
        return 'Unknown Breed'

    cleaned = str(breed_prediction).strip()
    cleaned = re.sub(r'\s*\((AI|Estimated offline|Estimated).*?\)\s*$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.replace('✓', '').strip()


def get_basic_breed_info(breed_prediction):
    """Return lightweight human-readable information for a predicted breed."""
    core_breed = _extract_core_breed_name(breed_prediction)
    normalized = core_breed.lower()

    for key, info in BREED_BASIC_INFO.items():
        if key in normalized:
            data = dict(info)
            data['breed_name'] = core_breed
            return data

    if 'buffalo' in normalized:
        return {
            'breed_name': core_breed,
            'animal_type': 'Buffalo',
            'origin': 'Indian subcontinent (varies by breed)',
            'primary_use': 'Mostly dairy and draught',
            'highlights': 'Typically robust build and well-suited to warm climates'
        }

    if any(term in normalized for term in ['cattle', 'cow', 'bull', 'ox', 'calf']):
        return {
            'breed_name': core_breed,
            'animal_type': 'Cattle',
            'origin': 'Indian and global breeding regions',
            'primary_use': 'Dairy, beef, or dual purpose depending on breed',
            'highlights': 'Body traits and milk potential vary across lines'
        }

    return {
        'breed_name': core_breed,
        'animal_type': 'Livestock',
        'origin': 'Not available',
        'primary_use': 'Not available',
        'highlights': 'Upload a clearer side profile for richer breed-level information'
    }


def get_breed_field_context(breed_prediction):
    """Return climate and management notes relevant to the predicted breed."""
    core_breed = _extract_core_breed_name(breed_prediction)
    normalized = core_breed.lower()

    for key, info in BREED_FIELD_CONTEXT.items():
        if key in normalized:
            return dict(info)

    if 'buffalo' in normalized:
        return {
            'climate_fit': 'Usually suited to warm regions when water access and heat relief are available.',
            'management_tip': 'Focus on hydration, shade, and body-condition support for stable production.'
        }

    if any(term in normalized for term in ['cattle', 'cow', 'bull', 'ox', 'calf']):
        return {
            'climate_fit': 'Suitability depends on whether the line is indigenous, crossbred, or exotic.',
            'management_tip': 'Match breeding decisions to local climate, fodder quality, and farm purpose.'
        }

    return {
        'climate_fit': 'A clearer breed prediction is needed before climate suitability can be stated confidently.',
        'management_tip': 'Use a cleaner side-view image to unlock more specific field guidance.'
    }


def get_goal_recommendation(breed_prediction, breeding_goal, breed_info, breeding_types):
    """Summarize how well the predicted breed fits the selected farm objective."""
    goal = _normalize_breeding_goal(breeding_goal)
    normalized_use = str(breed_info.get('primary_use', '')).lower()
    normalized_breed = _extract_core_breed_name(breed_prediction).lower()
    normalized_tags = ' '.join(breeding_types).lower()

    goal_labels = {
        'balanced': 'Balanced / General',
        'dairy': 'Dairy Focus',
        'draught': 'Draught / Work Focus',
        'dual': 'Dual-purpose Focus'
    }

    fit = 'moderate'
    summary = 'The current result looks usable for a mixed farm decision, but the image does not justify a hard recommendation.'

    if goal == 'dairy':
        if 'dairy' in normalized_use or 'milk' in normalized_tags:
            fit = 'strong'
            summary = 'This looks like a strong dairy-side match based on breed use and the current prediction signals.'
        elif 'dual' in normalized_use or 'dual-purpose' in normalized_tags:
            fit = 'moderate'
            summary = 'This can support dairy goals, but it is not as milk-specialized as a strong dairy line.'
        else:
            fit = 'limited'
            summary = 'This prediction is not strongly aligned with milk-first selection.'
    elif goal == 'draught':
        if 'draught' in normalized_use or 'work' in normalized_use or 'draught' in normalized_tags or 'rugged' in normalized_tags:
            fit = 'strong'
            summary = 'This result aligns well with field-work or structural-strength goals.'
        elif 'dual' in normalized_use or 'dual-purpose' in normalized_tags:
            fit = 'moderate'
            summary = 'This may work for practical field use, but it is not a dedicated draught-first line.'
        else:
            fit = 'limited'
            summary = 'This breed signal does not look ideal for work-focused breeding.'
    elif goal == 'dual':
        if 'dual' in normalized_use or 'dual-purpose' in normalized_tags:
            fit = 'strong'
            summary = 'This prediction is well aligned with farms balancing production and utility.'
        elif 'dairy' in normalized_use or 'draught' in normalized_use or 'work' in normalized_use:
            fit = 'moderate'
            summary = 'This can support a dual-purpose herd, but one side may dominate the breeding value.'
        else:
            fit = 'limited'
            summary = 'Dual-purpose suitability is not strongly supported by the current prediction.'
    else:
        if any(term in normalized_breed for term in ['unknown', 'livestock']):
            fit = 'limited'
            summary = 'The image produced a broad livestock result, so a balanced recommendation stays conservative.'
        else:
            fit = 'strong'
            summary = 'The breed signal appears usable for general herd planning and broad compatibility review.'

    fit_titles = {
        'strong': 'Strong fit',
        'moderate': 'Moderate fit',
        'limited': 'Limited fit'
    }

    return {
        'selected_goal_label': goal_labels[goal],
        'fit': fit,
        'fit_label': fit_titles[fit],
        'summary': summary
    }


def get_prediction_explanation(breed_prediction, measurements, classification_score):
    """Generate concise, human-readable reasons behind the current prediction."""
    notes = []
    breed_text = str(breed_prediction or '')

    if 'AI:' in breed_text:
        notes.append('The primary breed label came from the AI image model rather than measurement fallback alone.')
    elif 'offline' in breed_text.lower() or 'estimated' in breed_text.lower():
        notes.append('The breed label relied on measurement-based fallback because AI evidence was weak or unavailable.')

    length = float(measurements.get('body_length', 0))
    height = float(measurements.get('height_at_withers', 0))
    chest = float(measurements.get('chest_width', 0))
    aspect_ratio = float(measurements.get('aspect_ratio', 0))
    avg_brightness = float(measurements.get('avg_brightness', 0))

    if length > 850 or height > 500:
        notes.append('The body frame appears relatively large in the extracted contour, which supports larger dairy or draught lines.')
    elif length and height:
        notes.append('The frame size looks moderate, which fits mid-sized indigenous cattle or buffalo profiles better than extreme large-frame breeds.')

    if chest > 420:
        notes.append('A broader chest reading suggests stronger body capacity and field robustness.')
    elif chest > 0:
        notes.append('The chest width signal points to a lighter or more compact body build.')

    if aspect_ratio >= 1.45:
        notes.append('The length-to-height ratio suggests a longer body profile, which often supports dairy-oriented evaluation.')
    elif aspect_ratio > 0:
        notes.append('The contour ratio is more compact, which can indicate sturdier build traits rather than a long dairy frame.')

    if avg_brightness < 130:
        notes.append('The darker overall tone in the image increases the chance of buffalo-like visual matching.')
    elif avg_brightness > 145:
        notes.append('The brighter coat signal pushed the analysis toward cattle-like visual patterns.')

    if classification_score >= 3:
        notes.append('The rule-based conformation score is high, so the measurement profile is internally consistent.')
    else:
        notes.append('The rule-based conformation score is moderate, so this result should be treated as guidance rather than a breeding decision on its own.')

    return notes[:5]


def build_history_entry(result):
    """Create a compact serializable record for client-side recent analysis history."""
    if not result:
        return None

    recommendation = result.get('goal_recommendation', {})
    breed_info = result.get('breed_info', {})
    return {
        'breed': result.get('breed'),
        'animal_type': breed_info.get('animal_type'),
        'origin': breed_info.get('origin'),
        'goal': result.get('breeding_goal'),
        'fit_label': recommendation.get('fit_label'),
        'summary': recommendation.get('summary')
    }


def _normalize_breeding_goal(raw_goal):
    """Normalize user-selected breeding goal into a supported value."""
    if not raw_goal:
        return 'balanced'

    goal = str(raw_goal).strip().lower()
    aliases = {
        'milk': 'dairy',
        'dairy-focused': 'dairy',
        'work': 'draught',
        'draft': 'draught',
        'dual-purpose': 'dual',
        'both': 'dual',
        'all': 'balanced',
        'general': 'balanced'
    }
    goal = aliases.get(goal, goal)
    return goal if goal in VALID_BREEDING_GOALS else 'balanced'


def get_compatible_breeding_types(breed_prediction, breeding_goal='balanced'):
    """Return practical breeding-purpose compatibility tags tailored to user goal."""
    goal = _normalize_breeding_goal(breeding_goal)
    core_breed = _extract_core_breed_name(breed_prediction)
    normalized = core_breed.lower()

    # High-yield dairy lines
    dairy_lines = [
        'murrah', 'nili-ravi', 'surti', 'pandharpuri', 'mehsana',
        'sahiwal', 'gir', 'red sindhi', 'tharparkar', 'rathi',
        'holstein', 'jersey', 'vechur', 'punganur'
    ]

    # Draught / work lines
    draught_lines = ['ongole', 'hariana', 'hallikar', 'kangayam', 'khillari', 'bhadawari']

    # Balanced dual-purpose lines
    dual_lines = ['kankrej', 'hariana', 'tharparkar', 'bhadawari']

    tags = []

    if any(key in normalized for key in dairy_lines):
        tags.extend(['Dairy-focused', 'Milk-yield improvement'])

    if any(key in normalized for key in draught_lines):
        tags.extend(['Draught-focused', 'Rugged field performance'])

    if any(key in normalized for key in dual_lines):
        tags.extend(['Dual-purpose (milk + work)', 'Adaptability breeding'])

    if 'buffalo' in normalized:
        tags.append('Buffalo line breeding')
    if any(term in normalized for term in ['cattle', 'cow', 'bull', 'ox', 'calf']):
        tags.append('Cattle line breeding')

    if not tags:
        tags = ['General livestock improvement', 'Conformation-based selection']

    if goal == 'dairy':
        if any(key in normalized for key in dairy_lines):
            tags.insert(0, 'Recommended for selected goal: Dairy')
        else:
            tags.insert(0, 'Limited dairy-first compatibility')
    elif goal == 'draught':
        if any(key in normalized for key in draught_lines):
            tags.insert(0, 'Recommended for selected goal: Draught')
        else:
            tags.insert(0, 'Limited draught-first compatibility')
    elif goal == 'dual':
        if any(key in normalized for key in dual_lines):
            tags.insert(0, 'Recommended for selected goal: Dual-purpose')
        else:
            tags.insert(0, 'Moderate dual-purpose compatibility')
    else:
        tags.insert(0, 'Balanced breeding compatibility')

    # De-duplicate while preserving order.
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            unique_tags.append(tag)
            seen.add(tag)

    return unique_tags


@app.context_processor
def inject_runtime_flags():
    """Expose runtime flags (like token availability) to templates."""
    return {
        'hf_token_configured': bool(os.getenv('HF_API_TOKEN'))
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_goal = 'balanced'
    logger.info("Request method: %s", request.method)
    if request.method == 'POST':
        logger.info("Processing POST request")
        selected_goal = _normalize_breeding_goal(request.form.get('breeding_goal'))
        if 'image' not in request.files:
            logger.error("No image in request.files")
            return render_template('index.html', error='No image provided', selected_goal=selected_goal)
        
        file = request.files['image']
        if file.filename == '':
            logger.error("No filename")
            return render_template('index.html', error='No image selected', selected_goal=selected_goal)
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logger.info("Image saved to %s", filepath)
        
        try:
            # Process image
            measurements = process_image(filepath)
            if not measurements:
                return render_template('index.html', error='Could not detect animal in image. Please upload a clearer image.', selected_goal=selected_goal)
            logger.info("Measurements: %s", measurements)
            
            # Classify
            score = classify_animal(measurements)
            breed = classify_breed(measurements, filepath)
            breed_info = get_basic_breed_info(breed)
            breeding_types = get_compatible_breeding_types(breed, selected_goal)
            field_context = get_breed_field_context(breed)
            goal_recommendation = get_goal_recommendation(breed, selected_goal, breed_info, breeding_types)
            prediction_explanation = get_prediction_explanation(breed, measurements, score)

            if breed.startswith('No cattle or buffalo detected') or breed.startswith('Low-confidence analysis') or breed.startswith('AI service unavailable and image measurements are insufficient') or breed.startswith('AI service unavailable. Measurement-only estimate is not reliable'):
                return render_template('index.html', error=breed)

            logger.info("Score: %s, Breed: %s", score, breed)
            
            # Prepare data
            result = {
                'measurements': measurements,
                'classification_score': score,
                'breed': breed,
                'breed_info': breed_info,
                'breeding_types': breeding_types,
                'field_context': field_context,
                'goal_recommendation': goal_recommendation,
                'prediction_explanation': prediction_explanation,
                'breeding_goal': selected_goal,
                'image_path': filepath
            }
            
            # Mock BPA integration
            logger.info("Data ready for BPA integration: %s", result)
            
        except Exception as e:
            logger.error("Error processing image: %s", str(e))
            return render_template('index.html', error=f'Error processing image: {str(e)}', selected_goal=selected_goal)
    
    return render_template(
        'index.html',
        result=result,
        result_history_entry=build_history_entry(result),
        selected_goal=selected_goal
    )

@app.route('/api/classify', methods=['POST'])
def api_classify():
    selected_goal = _normalize_breeding_goal(request.form.get('breeding_goal'))
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        # Process image
        measurements = process_image(filepath)
        
        # Classify
        score = classify_animal(measurements)
        breed = classify_breed(measurements, filepath)
        breed_info = get_basic_breed_info(breed)
        breeding_types = get_compatible_breeding_types(breed, selected_goal)
        field_context = get_breed_field_context(breed)
        goal_recommendation = get_goal_recommendation(breed, selected_goal, breed_info, breeding_types)
        prediction_explanation = get_prediction_explanation(breed, measurements, score)

        if breed.startswith('No cattle or buffalo detected') or breed.startswith('Low-confidence analysis') or breed.startswith('AI service unavailable and image measurements are insufficient') or breed.startswith('AI service unavailable. Measurement-only estimate is not reliable'):
            return jsonify({'error': breed}), 400
        
        # Prepare data
        data = {
            'measurements': measurements,
            'classification_score': score,
            'breed': breed,
            'breed_info': breed_info,
            'breeding_types': breeding_types,
            'field_context': field_context,
            'goal_recommendation': goal_recommendation,
            'prediction_explanation': prediction_explanation,
            'breeding_goal': selected_goal,
            'image_path': filepath
        }
        
        # Mock BPA integration
        logger.info("Data ready for BPA integration: %s", data)
        
        return jsonify(data)
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        return jsonify({'error': 'Error processing image'}), 500


@app.route('/api/translate', methods=['POST'])
def api_translate():
    payload = request.get_json(silent=True) or {}
    target_lang = str(payload.get('target_lang', '')).strip()
    texts = payload.get('texts', [])

    if not target_lang:
        return jsonify({'error': 'target_lang is required'}), 400

    if not isinstance(texts, list):
        return jsonify({'error': 'texts must be an array'}), 400

    # Keep payload bounded for responsiveness and to avoid abuse.
    texts = texts[:200]

    # Normalize target language for translator endpoint.
    normalized_target_lang = target_lang.split('-')[0].strip().lower() or 'en'

    def translate_source(source):
        override = TRANSLATION_OVERRIDES.get(normalized_target_lang, {}).get(source)
        if override:
            return source, override

        try:
            url = (
                'https://translate.googleapis.com/translate_a/single'
                f'?client=gtx&sl=en&tl={quote(normalized_target_lang)}&dt=t&q={quote(source)}'
            )
            with urlopen(url, timeout=8) as response:
                raw = response.read().decode('utf-8', errors='replace')
                data = json.loads(raw)

            translated_chunks = []
            if isinstance(data, list) and data and isinstance(data[0], list):
                for chunk in data[0]:
                    if isinstance(chunk, list) and chunk:
                        translated_chunks.append(str(chunk[0] or ''))

            translated = ''.join(translated_chunks).strip()
            translated = re.sub(r'\s+', ' ', translated).strip()
            return source, translated if translated else source
        except Exception:
            return source, source

    translations = {}
    unique_sources = []
    seen_sources = set()
    for item in texts:
        source = str(item or '').strip()
        if not source or source in seen_sources:
            continue
        unique_sources.append(source)
        seen_sources.add(source)

    if unique_sources:
        worker_count = min(8, len(unique_sources))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(translate_source, source) for source in unique_sources]
            for future in as_completed(futures):
                source, translated = future.result()
                translations[source] = translated

    return jsonify({'translations': translations})

if __name__ == '__main__':
    app.run(debug=True, port=5500)