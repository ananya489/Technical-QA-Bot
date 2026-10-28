import json
from pathlib import Path

from src.pipeline import TechnicalQABot
from src.qa_model import ExtractiveQAModel
from config import DATA_DIR, RESULTS_DIR


def normalize(text):
    return ' '.join(text.lower().split())


def main():
    data = json.loads((DATA_DIR / 'domain_test.json').read_text(encoding='utf-8'))
    reader = ExtractiveQAModel()
    try:
        reader.load()
        mode = 'roberta'
    except Exception:
        mode = 'fallback'
    bot = TechnicalQABot(json.loads((DATA_DIR / 'university_manual.json').read_text()), reader=reader, use_model=False)
    exact = f1 = answered = 0
    rows = []
    for item in data:
        result = bot.ask(item['question'])
        gold = item.get('answers', item.get('answer', ''))
        if isinstance(gold, dict):
            gold = gold.get('text', [''])[0] if gold.get('text') else ''
        pred = result['answer']
        if pred.startswith('I couldn\'t find'):
            rows.append({'question': item['question'], 'prediction': pred, 'gold': gold})
            continue
        answered += 1
        exact += int(normalize(pred) == normalize(gold))
        p = set(normalize(pred).split()); g = set(normalize(gold).split())
        f1 += 0 if not p or not g else 2 * len(p & g) / (len(p) + len(g))
        rows.append({'question': item['question'], 'prediction': pred, 'gold': gold})
    n = len(data)
    print(f'Mode: {mode}')
    print(f'Exact Match: {exact / max(1, n):.3f}')
    print(f'F1: {f1 / max(1, n):.3f}')
    print(f'Answered: {answered}/{n}')
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / 'evaluation.json').write_text(json.dumps({'mode': mode, 'exact_match': exact / max(1,n), 'f1': f1 / max(1,n), 'answered': answered, 'total': n, 'details': rows}, indent=2))


if __name__ == '__main__':
    main()
