#!/usr/bin/env python

import logging
import sys
import os
import importlib

from blocks.extensions import FinishAfter
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent
from predict import PredictDataStream

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

import data
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    test_path = path = os.path.join(os.getenv("DATAPATH"),
                                    "deepmind-qa/cnn/questions/test")
    vocab_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt")

    ds_test, test_stream = data.setup_datastream(test_path, vocab_path, config)
    model_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds_test.vocab_size)

    # Build the Blocks stuff for training
    model = Model(m.sgd_cost)

    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters)
    extensions = []
    if config.save_freq is not None and model_path is not None:
        extensions += [
            SaveLoadParams(path=model_path,
                           model=model,
                           before_training=False,
                           after_training=False,
                           after_epoch=False,
                           every_n_batches=config.save_freq)
        ]
    extensions += [
        PredictDataStream(
            data_stream = test_stream,
            variables = [v for l in m.monitor_vars_test for v in l]
        ),
        ]

    extensions += [ FinishAfter(after_n_epochs=1)]

    main_loop = MainLoop(
        model=model,
        data_stream=test_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    # Run the model !
    main_loop.run()



#  vim: set sts=4 ts=4 sw=4 tw=0 et :
