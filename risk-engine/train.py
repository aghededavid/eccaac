from risk_engine.core.trainer import (
    SiameseTrainer,
    TrainingConfig,
    build_synthetic_dataset,
    BehavioralPairDataset
)

from risk_engine.core.siamese_network import (
    build_model,
    SiameseConfig,
)

import os

os.makedirs('/opt/eccaac/checkpoints', exist_ok=True)

cfg = TrainingConfig(
    checkpoint_dir='/opt/eccaac/checkpoints',
    epochs=30,
    batch_size=32,
)

model = build_model(SiameseConfig())

user_seqs, attack_seqs = build_synthetic_dataset(
    n_users=50,
    sequences_per_user=20
)

dataset = BehavioralPairDataset(
    user_seqs,
    attack_seqs,
    cfg=cfg
)

trainer = SiameseTrainer(model, cfg)

trainer.train(dataset)

print('Bootstrap training complete.')
print('Model saved to /opt/eccaac/checkpoints/best_model.pt')
