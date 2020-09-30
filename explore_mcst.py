# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     explore_mcst
   Description :
   Author :       chenhao
   date：          2020/9/9
-------------------------------------------------
   Change Activity:
                   2020/9/9:
-------------------------------------------------
"""
import sys
import logging
from ai_game.mcst import MCST, Node


def explore_node(node: Node):
    print(node.get_detail_str())
    while True:
        action = input("choose action")
        print(action)


if __name__ == '__main__':
    mcst_path = sys.argv[1]
    mcst = MCST.load(path=mcst_path, include_estimator=False)
    node = mcst.root
    explore_node(node)
