"""
Dataset Download & Sample Data Generation Script
Downloads datasets directly from Kaggle via the Kaggle API.
If Kaggle credentials are not yet configured, provides an automated fallback
generating realistic seed datasets for Malayalam, Hindi, Tanglish, and Romanized Kannada.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import RAW_DATA_DIR, KAGGLE_DATASETS

# Representative seed samples for offline testing / immediate execution
SYNTHETIC_SEEDS = {
    "malayalam": [
        ("ഈ സിനിമ വളരെ മനോഹരമാണ്! അഭിനയവും പാട്ടുകളും സൂപ്പർ.", "Positive"),
        ("കഥ വളരെ മോശമായിരുന്നു. സമയം പാഴായി.", "Negative"),
        ("സാധാരണ സിനിമ, വലിയ കുഴപ്പമില്ല.", "Neutral"),
        ("നല്ല സംവിധാനം, മികച്ച ദൃശ്യവിസ്മയം.", "Positive"),
        ("തീർത്തും നിരാശാജനകം. ഒരു പുതുമയും ഇല്ല.", "Negative"),
        ("ക്ലൈമാക്സ് നന്നായിരുന്നു എന്നാൽ തുടക്കം ലാഗ് ആയിരുന്നു.", "Neutral/Mixed"),
        ("താരങ്ങളുടെ പ്രകടനം ഉഗ്രൻ ആയിരുന്നു.", "Positive"),
        ("ബോറടിപ്പിച്ചു, പണം പോയി.", "Negative"),
        ("ഒറ്റത്തവണ കാണാം.", "Neutral"),
        ("സംഗീതം ഹൃദയസ്പർശിയായി തോന്നി.", "Positive")
    ],
    "hindi": [
        ("यह फिल्म बहुत ही शानदार और प्रेरणादायक है।", "Positive"),
        ("कहानी बहुत कमजोर और उबाऊ थी, बिल्कुल मजा नहीं आया।", "Negative"),
        ("औसत दर्जे की फिल्म है, एक बार देखी जा सकती है।", "Neutral"),
        ("एक्टिंग लाजवाब थी और डायलॉग्स भी दमदार थे।", "Positive"),
        ("पैसे की बर्बादी, कोई सिर-पैर नहीं था कहानी का।", "Negative"),
        ("पहला हाफ अच्छा था लेकिन सेकंड हाफ बहुत धीमा था।", "Neutral/Mixed"),
        ("निर्देशन और बैकग्राउंड म्यूजिक बहुत प्रभावशाली है।", "Positive"),
        ("बहुत ही घटिया निर्देशन और बेकार स्क्रीनप्ले।", "Negative"),
        ("समय काटने के लिए ठीक-ठाक फिल्म है।", "Neutral"),
        ("हर किसी को यह पारिवारिक सिनेमा देखना चाहिए।", "Positive")
    ],
    "tanglish": [
        ("Padam vera level bro, Rajini acting mass!", "Positive"),
        ("Romba mokka padam, time waste absolutely.", "Negative"),
        ("Average movie da, one time watchable.", "Neutral"),
        ("Anirudh bgm semma fire, theatres la vibe!", "Positive"),
        ("Story la perusa onnum illa, boring second half.", "Negative"),
        ("First half super but climax could have been better.", "Mixed_feelings"),
        ("Acting performance semma mass, directional brilliance.", "Positive"),
        ("Worst movie ever seen in recent times.", "Negative"),
        ("Normal entertainment film, nothing extraordinary.", "Neutral"),
        ("Family oda enjoy pannalam, super feel good.", "Positive")
    ],
    "kannada": [
        ("Cinema tumba chennagide, acting super aagide!", "Positive"),
        ("Kathe tumba bejar aaythu, time waste movie.", "Negative"),
        ("Normal movie, ond sala nodbahudu.", "Neutral"),
        ("Direction matthe background music adbhuta aagide.", "Positive"),
        ("Ketta screenplay, tumba bore hodedru.", "Negative"),
        ("First half chennagittu aadre climax isht aaglilla.", "Mixed_feelings"),
        ("Hero acting matthe comedy scene highlight.", "Positive"),
        ("Duddu waste guru, yaroo nodbedi.", "Negative"),
        ("Parvagilla, time pass ge okay.", "Neutral"),
        ("Ella family member galu nodalebekada movie.", "Positive")
    ]
}


def download_from_kaggle(dataset_slug: str, target_dir: Path) -> bool:
    """
    Invokes the Kaggle CLI to download and extract dataset.
    """
    print(f"Downloading {dataset_slug} to {target_dir}...")
    try:
        cmd = f"kaggle datasets download -d {dataset_slug} --unzip -p \"{target_dir}\""
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Successfully downloaded {dataset_slug}")
            return True
        else:
            print(f"Kaggle download notice: {res.stderr.strip() or res.stdout.strip()}")
            return False
    except Exception as e:
        print(f"Could not run Kaggle CLI: {e}")
        return False


def generate_seed_datasets(output_dir: Path, multiplier: int = 15):
    """
    Generates reproducible seed CSV files if Kaggle API keys are absent.
    """
    print(f"\nGenerating offline seed datasets in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang, samples in SYNTHETIC_SEEDS.items():
        expanded = []
        for rep in range(multiplier):
            for text, sent in samples:
                expanded.append((f"{text} [ref_{rep}_{sent}]", sent))
        df = pd.DataFrame(expanded, columns=["text", "sentiment"])
        out_path = output_dir / f"{lang}_raw.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Created seed dataset: {out_path} ({len(df)} samples)")


def setup_all_datasets():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_downloaded = True

    for lang, slug in KAGGLE_DATASETS.items():
        dest = RAW_DATA_DIR / lang
        dest.mkdir(parents=True, exist_ok=True)
        success = download_from_kaggle(slug, dest)
        if not success:
            all_downloaded = False

    if not all_downloaded:
        print("\nNote: One or more datasets could not be downloaded via Kaggle CLI (check kaggle.json).")
        print("Populating initial raw datasets so you can test and train immediately.")
        generate_seed_datasets(RAW_DATA_DIR)


if __name__ == "__main__":
    setup_all_datasets()
