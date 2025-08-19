def test_imports():
    from backend.src.config import Config
    from backend.src.pipeline import HybridLegalNLPPipeline
    assert Config
    assert HybridLegalNLPPipeline
