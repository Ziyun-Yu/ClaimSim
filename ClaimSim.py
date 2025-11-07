# from simcse import SimCSE
import time
import string
from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Tuple
import pandas as pd
import re
import json
import argparse
from jinja2 import Environment, FileSystemLoader
import concurrent.futures
from tqdm import tqdm
warnings.filterwarnings("ignore")

from utils.llm import gpt  # noqa: E402  (local import)

BASE_DIR = Path(__file__).resolve().parent


FINAL_SELECTED_FEATURES = {
    "Q260": "Sex",
    "Q261": "Year of birth",
    "X003R": "Age",
    "Q263": "Respondent immigrant",
    "Q264": "Mother immigrant",
    "Q265": "Father immigrant",
    "Q266": "Country of birth: Respondent",
    "Q267": "Country of birth: Mother of the respondent",
    "Q268": "Country of birth: Father of the respondent",
    "Q269": "Respondent citizen",
    "Q270": "Number of people in household",
    "Q271": "Do you live with your parents",
    "Q273": "Marital status",
    "Q274": "How many children do you have",
    "Q275R": "Highest educational level: Respondent",
    "Q276R": "Highest educational level: Respondent´s Spouse",
    "Q277R": "Highest educational level: Respondent´s Mother",
    "Q278R": "Highest educational level: Respondent´s Father",
    "Q279": "Employment status",
    "Q280": "Employment status - Respondent´s Spouse",
    "Q281": "Respondent - Occupational group",
    "Q282": "Respondent´s Spouse - Occupational group",
    "Q283": "Respondent´s Father - Occupational group (when respondent was 14 years old)",
    "Q284": "Sector of employment",
    "Q285": "Are you the chief wage earner in your house",
    "Q286": "Family savings during past year",
    "Q287": "Social class",
    "Q288R": "Income level",
    "Q289": "Religious denominations - major groups",
}


def clean_line(line: str) -> str:
    return re.sub(r"[*#-]", "", line).strip()


def extract_tagged_value(response: str, tag: str) -> str:
    pattern = rf"^{tag}:\s*(.*)$"
    for raw_line in response.splitlines():
        cleaned_line = clean_line(raw_line)
        match = re.match(pattern, cleaned_line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claim generation and evaluation pipeline."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="gender",
        help="Domain key used to index the question configuration.",
    )
    parser.add_argument(
        "--questions",
        nargs="*",
        help="Specific question IDs to run. Defaults to all questions in the domain config.",
    )
    parser.add_argument(
        "--question-config",
        type=Path,
        default=BASE_DIR / "data" / "question_config.json",
        help="Path to the question configuration JSON file.",
    )
    parser.add_argument(
        "--binary-config",
        type=Path,
        default=BASE_DIR / "data" / "question_binary_config.json",
        help="Path to the binary label configuration JSON file.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=BASE_DIR / "data" / "sampled_100_rows.csv",
        help="CSV file containing the survey data.",
    )
    parser.add_argument(
        "--demographic-info",
        type=Path,
        default=BASE_DIR / "data" / "demographic_info_less.json",
        help="JSON file with demographic metadata.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=BASE_DIR / "prompts",
        help="Directory containing Jinja template files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs",
        help="Directory where generated results will be saved.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of threads for processing rows.",
    )
    parser.add_argument(
        "--num-claims",
        type=int,
        default=5,
        help="Number of claims to sample per feature value.",
    )
    return parser.parse_args()


def load_templates(template_dir: Path) -> Tuple[Environment, Environment]:
    claim_env = Environment(loader=FileSystemLoader(str(template_dir)))
    claim_env.globals.update(enumerate=enumerate)

    generation_env = Environment(loader=FileSystemLoader(str(template_dir)))
    generation_env.globals.update(enumerate=enumerate)

    return claim_env, generation_env


def render_claims(
    claim_env: Environment,
    template_name: str,
    feature_category: str,
    feature_label: str,
    instruction: str,
    question: str,
    num_claims: int,
) -> List[str]:
    template = claim_env.get_template(template_name)
    claims: List[str] = []

    for _ in range(num_claims):
        prompt = template.render(
            feature_category=feature_category,
            feature_label=feature_label,
            instruction=instruction,
            question=question,
        )
        claim = ""
        while not claim:
            response = gpt(prompt, stop=None)
            claim = extract_tagged_value(response, "claim")
        claims.append(claim)

    return claims


def render_summary(
    claim_env: Environment,
    template_name: str,
    feature_category: str,
    feature_label: str,
    claims: List[str],
) -> str:
    template = claim_env.get_template(template_name)
    prompt = template.render(
        feature_category=feature_category,
        feature_label=feature_label,
        claims=claims,
    )
    summary = ""
    while not summary:
        response = gpt(prompt, stop=["\n"])
        summary = extract_tagged_value(response, "summary")
    return summary


def generate_claims(
    row: pd.Series,
    question_id: str,
    domain: str,
    num_claims: int,
    claim_env: Environment,
    question_config: Dict,
    features: Iterable[str],
    cache: Dict[str, Tuple[str, List[str]]],
) -> Dict[str, Tuple[str, str, List[str]]]:
    extracted_claims: Dict[str, Tuple[str, str, List[str]]] = {}

    for feature_key in features:
        label = str(row[feature_key])
        cache_key = f"{feature_key}:{label}"

        if cache_key not in cache:
            claims = render_claims(
                claim_env=claim_env,
                template_name="claims.md",
                feature_category=FINAL_SELECTED_FEATURES[feature_key],
                feature_label=label,
                instruction=question_config[domain]["test"][question_id]["instruction"],
                question=question_config[domain]["test"][question_id]["question"],
                num_claims=num_claims,
            )

            summary = render_summary(
                claim_env=claim_env,
                template_name="summary.md",
                feature_category=FINAL_SELECTED_FEATURES[feature_key],
                feature_label=label,
                claims=claims,
            )

            cache[cache_key] = (summary, claims)

        summary_text, stored_claims = cache[cache_key]
        extracted_claims[summary_text] = (
            FINAL_SELECTED_FEATURES[feature_key],
            label,
            stored_claims,
        )

    return extracted_claims


def run_generation(
    question_id: str,
    row_index: int,
    row: pd.Series,
    domain: str,
    claim_env: Environment,
    generation_env: Environment,
    question_config: Dict,
    num_claims: int,
    features: Iterable[str],
    cache: Dict[str, Tuple[str, List[str]]],
) -> Dict[str, object]:
    extracted_claims = generate_claims(
        row=row,
        question_id=question_id,
        domain=domain,
        num_claims=num_claims,
        claim_env=claim_env,
        question_config=question_config,
        features=features,
        cache=cache,
    )

    demo_infos = {
        FINAL_SELECTED_FEATURES[key]: row[key] for key in FINAL_SELECTED_FEATURES
    }

    template = generation_env.get_template("generation.md")
    target = ""
    prompt = ""
    response = ""

    while not target:
        prompt = template.render(
            demo_infos=demo_infos,
            instruction=question_config[domain]["test"][question_id]["instruction"],
            question=question_config[domain]["test"][question_id]["question"],
            labels=question_config[domain]["test"][question_id]["labels"],
            claims=extracted_claims,
        )

        response = gpt(prompt, stop=None).lower()
        lines = response.splitlines()

        for ans in question_config[domain]["test"][question_id]["labels"]:
            if lines and ans.lower() in lines[0] and len(target) < len(ans):
                target = ans

    labels = question_config[domain]["test"][question_id]["labels"]
    raw_answer = str(row[question_id])
    true_label = "Missing"

    try:
        answer_index = int(raw_answer)
        if question_id not in {"Q94", "Q128"}:
            answer_index -= 1
        if 0 <= answer_index < len(labels):
            true_label = labels[answer_index]
    except ValueError:
        true_label = "Missing"

    return {
        "pred": target,
        "true": true_label,
        "record": {
            "question_id": question_id,
            "row_index": row_index,
            "prompt": prompt,
            "response": response,
            "predicted_label": target,
            "true_label": true_label,
            "extracted_claims": extracted_claims,
        },
    }


def evaluate_predictions(
    predictions: List[str],
    truths: List[str],
    binary_predictions: List[str],
    binary_truths: List[str],
) -> Tuple[float, float]:
    exact_correct = sum(p.lower() == t.lower() for p, t in zip(predictions, truths))
    binary_correct = sum(
        p.lower() == t.lower() if t != "Missing" else p.lower() == t.lower()
        for p, t in zip(binary_predictions, binary_truths)
    )

    exact_accuracy = exact_correct / len(truths) if truths else 0.0
    binary_accuracy = binary_correct / len(binary_truths) if binary_truths else 0.0
    return exact_accuracy, binary_accuracy


def main() -> None:
    args = parse_args()

    question_config = pd.read_json(args.question_config)
    with args.demographic_info.open("r", encoding="utf-8") as f:
        demographic_info = json.load(f)
    df = pd.read_csv(args.data_file)
    with args.binary_config.open("r", encoding="utf-8") as f:
        binary_config = json.load(f)

    questions = args.questions
    if not questions:
        questions = list(question_config[args.domain]["test"].keys())

    claim_env, generation_env = load_templates(args.template_dir)
    output_dir = args.output_dir / args.domain
    ensure_dir(output_dir)

    results_summary: List[Dict[str, float]] = []

    for question_id in questions:
        cache: Dict[str, Tuple[str, List[str]]] = {}
        preds: List[str] = []
        truths: List[str] = []
        binary_preds: List[str] = []
        binary_truths: List[str] = []
        records: List[Dict[str, object]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_generation,
                    question_id,
                    idx,
                    row,
                    args.domain,
                    claim_env,
                    generation_env,
                    question_config,
                    args.num_claims,
                    demographic_info.keys(),
                    cache,
                )
                for idx, row in df.iterrows()
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                preds.append(result["pred"])
                truths.append(result["true"])

                binary_preds.append(binary_config[args.domain][question_id][result["pred"]])
                if result["true"] != "Missing":
                    binary_truths.append(binary_config[args.domain][question_id][result["true"]])
                else:
                    binary_truths.append("Missing")
                records.append(result["record"])

        exact_accuracy, binary_accuracy = evaluate_predictions(
            preds, truths, binary_preds, binary_truths
        )

        results_summary.append(
            {
                "Question": question_id,
                "Exact Accuracy": exact_accuracy,
                "Binary Accuracy": binary_accuracy,
            }
        )

        output_path = output_dir / f"{question_id}_results.json"
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"{'Question':<12}{'Exact Accuracy':<18}{'Binary Accuracy':<18}")
        for item in results_summary:
            print(
                f"{item['Question']:<12}"
                f"{item['Exact Accuracy']:<18.4f}"
                f"{item['Binary Accuracy']:<18.4f}"
            )

    print(f"{'Question':<12}{'Exact Accuracy':<18}{'Binary Accuracy':<18}")
    for item in results_summary:
        print(
            f"{item['Question']:<12}"
            f"{item['Exact Accuracy']:<18.4f}"
            f"{item['Binary Accuracy']:<18.4f}"
        )


if __name__ == "__main__":
    main()