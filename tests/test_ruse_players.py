"""tests/test_ruse_players.py — scripted players, the LLM adapter, and Aeon at
the table (with a tiny randomly initialised checkpoint when torch is present)."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.ruse import (Match, RuseConfig, default_board, HeuristicPlayer,
                       RandomPlayer, LLMPlayer, make_player, parse_orders)
from aeon.ruse.players import SYSTEM_PROMPT


def test_scripted_players_emit_parseable_orders():
    m = Match(default_board(), seed=0)
    for P in (HeuristicPlayer, RandomPlayer):
        p = P("CAPITAL", seed=3)
        obs = m.observe("CAPITAL")
        text = p.act(obs, m.render("CAPITAL"))
        parsed = parse_orders(text)
        assert parsed.orders, f"{P.__name__} produced nothing parseable: {text!r}"


def test_heuristic_hardens_in_crisis():
    m = Match(default_board(), seed=0)
    m.dyn["CAPITAL"].x[:] = 0.2       # deep stress -> lambda_max > 0
    obs = m.observe("CAPITAL")
    assert obs.lambda_max > 0
    text = HeuristicPlayer("CAPITAL", seed=0).act(obs, m.render("CAPITAL"))
    assert text.splitlines()[0].startswith("HARDEN")


def test_llm_player_parses_messy_reply_and_keeps_history():
    replies = iter([
        "Thinking...</think>\n```\n1. BUILD credit RESERVE 0.4\n- ruse decoy COURT\nnonsense line\n```",
        "",                       # empty reply -> PASS
    ])
    seen = []

    def gen(messages):
        seen.append(messages)
        return next(replies)

    p = LLMPlayer("CAPITAL", gen, history_turns=1)
    m = Match(default_board(), seed=0)
    out1 = p.act(m.observe("CAPITAL"), m.render("CAPITAL"))
    assert out1 == "BUILD credit RESERVE 0.40\nRUSE DECOY COURT"
    m.step({"CAPITAL": out1, "INDUSTRIAL": "PASS"})
    out2 = p.act(m.observe("CAPITAL"), m.render("CAPITAL"))
    assert out2 == "PASS"
    assert seen[0][0]["role"] == "system" and seen[0][0]["content"] == SYSTEM_PROMPT
    # second call carries the previous (user, assistant) pair
    assert [m_["role"] for m_ in seen[1]] == ["system", "user", "assistant", "user"]
    assert seen[1][2]["content"] == out1
    assert len(p.transcript) == 2


def test_llm_player_survives_generator_failure():
    def boom(messages):
        raise RuntimeError("no GPU today")
    p = LLMPlayer("CAPITAL", boom)
    m = Match(default_board(), seed=0)
    assert p.act(m.observe("CAPITAL"), m.render("CAPITAL")) == "PASS"
    assert p.parse_errors and "generate failed" in p.parse_errors[0]


def test_llm_player_completes_a_match():
    p = LLMPlayer("INDUSTRIAL", lambda msgs: "BUILD law COUNCIL 0.3\nHARDEN FINANCE")
    m = Match(default_board(), seed=1)
    res = m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=1), "INDUSTRIAL": p}, max_turns=5)
    assert res.turns == 5 and len(p.transcript) == 5


def test_make_player():
    assert isinstance(make_player("random", "CAPITAL"), RandomPlayer)
    assert isinstance(make_player("heuristic", "CAPITAL"), HeuristicPlayer)
    with pytest.raises(ValueError):
        make_player("aeon", "CAPITAL")          # needs a checkpoint
    with pytest.raises(ValueError):
        make_player("chess-engine", "CAPITAL")


# ---- Aeon itself, on a tiny random checkpoint ------------------------------------

def _tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from aeon.config import AeonConfig
    from aeon.model import AeonR1ForCausalLM

    words = ["<pad>", "<eos>", "BUILD", "RUSE", "PASS", "HARDEN", "law", "credit",
             "COURT", "RESERVE", "FINANCE", "DECOY", "0.4", "0.3", "<unk>"]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="<eos>")
    cfg = AeonConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=512, h_rec=16, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = AeonR1ForCausalLM(cfg)
    ck = tmp_path / "tiny_aeon"
    model.save_pretrained(ck)
    tok.save_pretrained(ck)
    return str(ck)


def test_aeon_player_plays_and_carries_state(tmp_path):
    ck = _tiny_checkpoint(tmp_path)
    from aeon.ruse.players import AeonPlayer
    p = AeonPlayer("INDUSTRIAL", ck, max_new_tokens=8, temperature=0.0,
                   device="cpu", dtype="float32")
    m = Match(default_board(), seed=2)
    n0 = p.recursion_norms()
    res = m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=1), "INDUSTRIAL": p}, max_turns=2)
    assert res.turns == 2 and len(p.transcript) == 2
    n1 = p.recursion_norms()
    assert n1["r"] != n0["r"]            # the state moved and persisted
    aud = p.audit()
    assert aud["chart_A_holds"]          # contraction certificate still holds
