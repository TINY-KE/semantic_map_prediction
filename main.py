
""" Entry point for training
"""

from train_options import TrainOptions
from mp_trainer import MP_TrainerAM
from tester import SemMapTester, SemMapSLAMer
import multiprocessing as mp


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = TrainOptions().parse_args()

    if options.is_train:
        trainer = MP_TrainerAM(options)
        trainer.train()

    else:
        if options.sem_map_test:
            tester = SemMapTester(options)
            print("     [zhjd-debug] Testing semantic map prediction...")
            tester.test_semantic_map()
        else:
            slamer = SemMapSLAMer(options)
            print("     [zhjd-debug] SLAM semantic map prediction...")
            slamer.test_semantic_map()
