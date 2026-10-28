import argparse
import json
from pathlib import Path

from src.pipeline import TechnicalQABot
from src.qa_model import ExtractiveQAModel
from config import DATA_DIR


def load_manual(path=None):
    p = Path(path) if path else DATA_DIR / 'university_manual.json'
    return json.loads(p.read_text(encoding='utf-8'))


def build_bot(manual, force_fallback=False):
    reader = ExtractiveQAModel()
    if not force_fallback:
        try:
            reader.load()
            print('Reader: RoBERTa extractive QA')
        except Exception as exc:
            print(f'Reader model unavailable ({type(exc).__name__}); using lightweight fallback.')
    else:
        print('Reader: lightweight fallback')
    return TechnicalQABot(manual, reader=reader, use_model=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--fallback', action='store_true')
    parser.add_argument('--manual', default=None)
    args = parser.parse_args()
    bot = build_bot(load_manual(args.manual), args.fallback)
    if args.demo:
        questions = [
            'What is the minimum attendance required for end semester exams?',
            'How many books can a student borrow?',
            'What are the computer lab timings?',
            'What is the swimming pool schedule?',
        ]
        for q in questions:
            print(f'\nQ: {q}\nA: {bot.ask(q)["answer"]}')
        return
    print("Technical Documentation QA Bot. Type 'sections' or 'exit'.")
    while True:
        q = input('\nYou: ').strip()
        if q.lower() == 'exit':
            break
        if q.lower() == 'sections':
            print('\n'.join(bot.retriever.names))
            continue
        result = bot.ask(q)
        print('Bot:', result['answer'])
        if result['section']:
            print('Source:', result['section'])


if __name__ == '__main__':
    main()
