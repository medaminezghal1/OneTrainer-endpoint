from util.import_util import script_imports

script_imports()

import json

from modules.util.config.TrainConfig import TrainConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.trainer.GenericTrainer import GenericTrainer


def main(config_path):
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main("default_config.json")  # Default config path
