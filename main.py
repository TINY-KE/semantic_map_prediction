
""" Entry point for training
"""

from train_options import TrainOptions
from mp_trainer import MP_TrainerAM
from tester import SemMapTester, SemMapSLAMer
from Searcher import SearchTester, SearchSLAMer
from Roser import RosTester
import multiprocessing as mp


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = TrainOptions().parse_args()

    if options.is_train:
        trainer = MP_TrainerAM(options)
        trainer.train()

    elif options.is_ros:
        import rospy
        rospy.init_node('semantic_map_prediction_node')
        roser = RosTester(options)
        print("\n\n     [zhjd-debug] ROS semantic map prediction, 等待ROS信息...")
        rospy.spin()

    else:
        if options.sem_map_test:
            # tester = SemMapTester(options)
            tester = SearchTester(options)
            print("     [zhjd-debug] Testing semantic map prediction...")
            tester.test_semantic_map()

        else:
            # slamer = SemMapSLAMer(options)
            slamer = SearchSLAMer(options)
            print("     [zhjd-debug] SLAM semantic map prediction...")
            slamer.test_semantic_map()
