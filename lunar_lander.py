"""Lander simulator with desired thruster activation
"""
import sys
import math
import random
import torch
from neural_network import Perceptron
from training_data import parse_training_data

LUNAR_GRAVITY = -1.62
ROCKET = 2.4

class Lander:
    def __init__(self):
        self.perceptron = Perceptron(2, 8, 8, 1)
        self.y = 100 # meters
        self.dy = 0 # m/s
        self.rocket = False

    def tick(self, dt):
        # act
        ddy = LUNAR_GRAVITY + (ROCKET if self.rocket else 0.0)
        self.y += self.dy * dt + ddy * dt**2 / 2.0
        self.dy += ddy * dt
        print(self)
        # think
        activation = self.perceptron.forward([self.y, self.dy])[0]
        self.rocket = activation > 0

    def fly(self, name) :
        print('fly', name)
        self.y = 80
        self.dy = 0
        self.rocket = False
        for t in range(30):
            self.tick(1)
            if self.y < 0 or self.y > 120 :
                break
        return self.y <= 0.0 and abs(self.dy) <= 1.0

    def train_step(self, training_data, desired_output):
        step_size = 0.001
        num_steps = 1
        loss_threshold = 20
        loss = self.perceptron.train_network(training_data, desired_output, step_size, num_steps, loss_threshold)
        print(f'loss {loss.item():0.4f}')
        print('perceptron', self.perceptron)

    def __str__(self):
        out = [f'lander y {self.y} dy {self.dy} rocket {self.rocket}']
        # out.append(str(self.perceptron))
        return '\n'.join(out)
    
    def print_rocket(self):
        out = []
        out.append(' '.join(f'{dy:10d}' for dy in range(-20, 21, 4)))
        for y in range(100, -1, -10):
            row = [f'{y:3d}']
            for dy in range(-20, 21, 4) :
                rocket = lander.perceptron.forward([y,dy])[0].item()
                row.append(f'{rocket:9.4f} ')
            out.append(' '.join(row))
        return '\n'.join(out)

if __name__ == '__main__':
    # random.seed(2024)
    training_data, desired_output = parse_training_data()
    lander = Lander()
    lander.fly(0)
    loss = lander.perceptron.train_network(training_data, desired_output, 0.01, 100, 10)
    print('loss', loss)
    print(lander.print_rocket())
    lander.fly(1)
