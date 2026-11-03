def test_imports():
    from risk_engine.core.trainer import SiameseTrainer
    from risk_engine.core.siamese_network import SiameseNetwork

    assert SiameseTrainer is not None
    assert SiameseNetwork is not None