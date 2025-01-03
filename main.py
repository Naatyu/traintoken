#!/usr/bin/env python
import os

os.environ["TRITON_INTERPRET"] = "1"
from traintoken import TrainToken

if __name__ == "__main__":
    test_texts = [
        "Hi, what's up ?",
        "Moi ca va très bien et toi ?",
        "I really like golf, especially McIlroy",
    ]

    # Create trainer
    trainer = TrainToken(max_vocab_size=300)
    trainer.train(test_texts, n_proc=os.cpu_count() - 1)
    # trainer.save_encoding
    print("Done")
