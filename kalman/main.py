#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:56:05 2021

@author: ainur
"""
from argparse import ArgumentParser
import random
import time

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

import matplotlib.pyplot as plt

TIME_SLEEP = 0.02
APP_NAME = "simple kalman"

keep_ploting = True


def on_key(event):
    global keep_ploting
    keep_ploting = False


def process_eval():
    plt.ion()

    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([[2.0], [0.0]])  # position  # velocity
    f.x = np.array([2.0, 0.0])
    f.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    f.H = np.array([[1.0, 0.0]])
    f.P *= 1000.0
    f.R = np.array([[5.0]])

    f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=2.13)
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_key)

    maxSize = 200
    # Установка отображаемых интервалов по осям
    ax.set_xlim(0, maxSize)
    ax.set_ylim(-10, 10)

    # Отобразить график фукнции в начальный момент времени
    x = np.arange(maxSize)
    y_true = np.zeros(maxSize, dtype=float)
    y_pred = np.zeros(maxSize, dtype=float)

    (line,) = ax.plot(x, y_true, label="True")
    (line2,) = ax.plot(x, y_pred, label="Pred")
    plt.legend(loc="best")

    try:
        while keep_ploting:
            z = random.uniform(-2, 2)

            y_true = np.append(y_true, z)[1:]
            line.set_ydata(y_true)

            y_pred = np.append(y_pred, f.x[0])[1:]
            line2.set_ydata(y_pred)
            # Отобразить новые данные
            fig.canvas.draw()
            fig.canvas.flush_events()
            f.predict()
            f.update(z)
            time.sleep(TIME_SLEEP)
        plt.ioff()
        plt.close()

    except KeyboardInterrupt:
        print("Closed Figure!")
        pass


def callback_eval(arguments):
    return process_eval()


def setup_parser(parser):
    """The function to setup parser arguments"""

    parser.set_defaults(callback=callback_eval)


def main():
    """Main module function"""
    parser = ArgumentParser(
        prog=APP_NAME,
        description="A tool to view kalman",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
