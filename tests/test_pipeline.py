import json
from pathlib import Path

from src.pipeline import TechnicalQABot, ABSTAIN
from src.qa_model import ExtractiveQAModel


ROOT = Path(__file__).resolve().parents[1]


def manual():
    return json.loads((ROOT / 'data' / 'university_manual.json').read_text(encoding='utf-8'))


def test_manual_loads():
    data = manual()
    assert len(data) >= 20
    assert 'attendance' in data


def test_retriever_finds_attendance():
    bot = TechnicalQABot(manual(), reader=ExtractiveQAModel(), use_model=False)
    result = bot.ask('What is the minimum attendance required?')
    assert result['section'] == 'attendance'


def test_empty_question():
    bot = TechnicalQABot(manual(), reader=ExtractiveQAModel(), use_model=False)
    assert 'Please enter' in bot.ask('')['answer']


def test_answerable_question_uses_manual():
    bot = TechnicalQABot(manual(), reader=ExtractiveQAModel(), use_model=False)
    result = bot.ask('How many books can a student borrow?')
    assert result['section'] == 'library'
    assert '4' in result['answer']


def test_unrelated_question_abstains():
    bot = TechnicalQABot(manual(), reader=ExtractiveQAModel(), use_model=False)
    result = bot.ask('Who won the football world cup in 2018?')
    assert result['answer'] == ABSTAIN
