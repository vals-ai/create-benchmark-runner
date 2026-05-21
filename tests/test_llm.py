from benchmark_runner.llm import build_llm_config


def test_default_and_sampling_config():
    cfg = build_llm_config()
    assert cfg.max_tokens is None
    assert cfg.temperature is None
    assert cfg.custom_endpoint is None

    cfg = build_llm_config(max_tokens=2048, temperature=0.7, top_p=0.9, top_k=40, reasoning_effort="high")
    assert cfg.max_tokens == 2048
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.9
    assert cfg.top_k == 40
    assert cfg.reasoning_effort == "high"


def test_provider_config_knobs():
    cfg = build_llm_config(custom_endpoint="http://my-proxy", custom_api_key="sk-xyz")
    assert cfg.custom_endpoint == "http://my-proxy"
    assert cfg.custom_api_key is not None
    assert cfg.custom_api_key.get_secret_value() == "sk-xyz"

    cfg = build_llm_config(chat_completions=True, disable_streaming=True)
    assert cfg.native is False
    assert cfg.provider_config is not None
    assert getattr(cfg.provider_config, "stream_completions") is False
