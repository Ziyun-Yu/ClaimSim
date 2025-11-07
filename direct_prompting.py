import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
import warnings
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
import concurrent.futures

from utils.llm import gpt 

warnings.filterwarnings("ignore")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct prompting baseline evaluation.")
    parser.add_argument("--domain", type=str, default="gender", help="Target domain key.")
    parser.add_argument(
        "--questions",
        nargs="*",
        help="Subset of question IDs. Defaults to all questions for domain.",
    )
    parser.add_argument(
        "--question-config",
        type=Path,
        default=BASE_DIR / "data" / "question_config.json",
        help="Path to question configuration JSON.",
    )
    parser.add_argument(
        "--binary-config",
        type=Path,
        default=BASE_DIR / "data" / "question_binary_config.json",
        help="Path to binary label configuration JSON.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=BASE_DIR / "data" / "sampled_100_rows.csv",
        help="CSV file with survey responses.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=BASE_DIR / "prompts",
        help="Directory containing direct prompting templates.",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default="direct_prompting_template.md",
        help="Template filename for direct prompting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "direct_prompting",
        help="Directory for saving model outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size for parallel execution.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_template_environment(template_dir: Path) -> Environment:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    env.globals.update(enumerate=enumerate)
    return env


def resolve_true_label(
    domain: str, question_id: str, row_value: str, question_config: Dict
) -> str:
    labels = question_config[domain]["test"][question_id]["labels"]
    try:
        idx = int(row_value)
        if question_id not in {"Q94", "Q128"}:
            idx -= 1
        if 0 <= idx < len(labels):
            return labels[idx]
    except ValueError:
        pass
    return "Missing"


def process_one_prompt(
    question_id: str,
    index: int,
    row: pd.Series,
    template,
    domain: str,
    question_config: Dict,
) -> Dict[str, object]:
    demo_infos = {FINAL_SELECTED_FEATURES[key]: row[key] for key in FINAL_SELECTED_FEATURES}

    target = ""
    prompt = ""
    response = ""

    while not target:
        prompt = template.render(
            demo_infos=demo_infos,
            instruction=question_config[domain]["test"][question_id]["instruction"],
            question=question_config[domain]["test"][question_id]["question"],
            labels=question_config[domain]["test"][question_id]["labels"],
            domain=domain,
        )
        response = gpt(prompt, stop=None).lower()
        lines = response.splitlines()
        if lines:
            for answer in question_config[domain]["test"][question_id]["labels"]:
                if answer.lower() in lines[0] and len(target) < len(answer):
                    target = answer

    true_label = resolve_true_label(domain, question_id, str(row[question_id]), question_config)

    return {
        "pred": target,
        "true": true_label,
        "record": {
            "question_id": question_id,
            "row_index": index,
            "prompt": prompt,
            "response": response,
            "predicted_label": target,
            "true_label": true_label,
        },
    }


def evaluate(preds: List[str], truths: List[str]) -> float:
    valid_pairs = [(p, t) for p, t in zip(preds, truths) if t != "Missing"]
    if not valid_pairs:
        return 0.0
    correct = sum(p.lower() == t.lower() for p, t in valid_pairs)
    return correct / len(valid_pairs)


def main() -> None:
    args = parse_args()

    question_config = pd.read_json(args.question_config)
    with args.binary_config.open("r", encoding="utf-8") as f:
        binary_config = json.load(f)
    df = pd.read_csv(args.data_file)

    questions = args.questions or list(question_config[args.domain]["test"].keys())
    template_env = load_template_environment(args.template_dir)
    template = template_env.get_template(args.template_name)
    ensure_dir(args.output_dir / args.domain)

    results_summary: List[Dict[str, float]] = []

    for question_id in questions:
        preds: List[str] = []
        truths: List[str] = []
        binary_preds: List[str] = []
        binary_truths: List[str] = []
        records: List[Dict[str, object]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    process_one_prompt,
                    question_id,
                    idx,
                    row,
                    template,
                    args.domain,
                    question_config,
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

        exact_accuracy = evaluate(preds, truths)
        binary_accuracy = evaluate(binary_preds, binary_truths)

        results_summary.append(
            {
                "Question": question_id,
                "Exact Accuracy": exact_accuracy,
                "Binary Accuracy": binary_accuracy,
            }
        )

        output_path = args.output_dir / args.domain / f"{question_id}_results.json"
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


if __name__ == "__main__":
    main()